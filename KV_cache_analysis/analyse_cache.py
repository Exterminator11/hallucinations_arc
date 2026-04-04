"""
Generic KV-cache hallucination analysis.
Works with any TransformerLens model — GQA or MHA, any depth, any head count.

Usage:
    python analysis_generic.py \
        --file  path/to/records.pt \
        --model my-model-name \
        --outdir ./results
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to labelled .pt records file")
    p.add_argument(
        "--model", required=True, help="Model name (used for output filenames)"
    )
    p.add_argument("--outdir", default=".", help="Directory to save plots")
    p.add_argument(
        "--label-key",
        default="hallucination_label",
        help="Key inside metadata dict that holds 0/1 hallucination label",
    )
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────


def load_data(file_path, label_key):
    print(f"Loading {file_path} ...")
    records = torch.load(file_path, weights_only=False)
    df = pd.DataFrame(records)

    # Drop records where the judge failed to produce a label
    before = len(df)
    df = df[df["metadata"].apply(lambda m: m.get(label_key, -1) != -1)].reset_index(
        drop=True
    )
    print(f"  {before} records loaded, {len(df)} usable after dropping parse failures.")

    labels = df["metadata"].apply(lambda m: m[label_key])
    print(f"  Hallucinated : {(labels == 1).sum()}")
    print(f"  Truthful     : {(labels == 0).sum()}")
    return df, label_key


# ── Helpers ────────────────────────────────────────────────────────────────────


def split_populations(df, label_key):
    mask = df["metadata"].apply(lambda m: m[label_key])
    return df[mask == 0], df[mask == 1]


def sorted_layers(df):
    return sorted(df.iloc[0]["layers"].keys())


def savefig(fig, outdir, model, suffix):
    safe_model = model.replace("/", "_").replace(" ", "_")
    path = os.path.join(outdir, f"{safe_model}_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def expand_kv_to_q_heads(v, n_q_heads):
    """
    Expand KV heads to match query heads for GQA models.
    v shape: [n_kv_heads, seq_len, head_dim]
    Returns: [n_q_heads, seq_len, head_dim]
    """
    n_kv_heads = v.shape[0]
    if n_kv_heads == n_q_heads:
        return v
    group_size = n_q_heads // n_kv_heads
    return v.repeat_interleave(group_size, dim=0)


def get_n_q_heads(df):
    """Derive number of query heads from pattern tensor shape."""
    sample_layer = sorted_layers(df)[0]
    pattern = df.iloc[0]["layers"][sample_layer]["pattern"]
    # pattern shape: [1, n_q_heads, seq_len_dst, seq_len_src]
    return pattern.shape[1]


def get_n_kv_heads(df):
    """Derive number of KV heads from value tensor shape."""
    sample_layer = sorted_layers(df)[0]
    v = df.iloc[0]["layers"][sample_layer]["v"]
    # v shape: [1, seq_len, n_kv_heads, head_dim]
    return v.shape[2]


def late_layer_cutoff(all_layers, fraction=0.7):
    """Return layers in the last (1 - fraction) of the network."""
    idx = int(len(all_layers) * fraction)
    return all_layers[idx:]


# ── Experiment 2 & 3: Key / Value norm trajectory ─────────────────────────────


def key_value_norm_trajectory(df, label_key, outdir, model):
    """
    Experiment 2 & 3 — Key & Value Norm Trajectory.

    Compute mean key and value norm at answer positions only (tokens after
    question_len), averaged across heads, per layer, comparing hallucinated
    vs. truthful populations.

    Interpretation:
        1. Both populations diverge at a specific layer → that layer is where
           the failure begins.
        2. Hallucinated norms are lower from layer 0 → collapse is inherited
           from embeddings; attention is not the cause.
        3. Similar throughout → key/value space does not reflect the hidden
           state collapse; look elsewhere (entropy, cancellation).
        4. Value norms track key norms closely → both collapse together,
           pointing to the same upstream failure.
        5. Value norms diverge where key norms did not → failure is in what
           gets written into the residual, not in what gets attended to.
    """
    print("\n[Exp 2/3] Key & Value norm trajectory ...")
    truth_df, halluc_df = split_populations(df, label_key)
    layers = sorted_layers(df)

    results = {}
    for label, subset in [("truth", truth_df), ("hallucinated", halluc_df)]:
        layer_norms = {"k": {}, "v": {}}
        for _, row in subset.iterrows():
            q_len = row["metadata"]["question_len"]
            for layer in layers:
                tensors = row["layers"][layer]
                for kv_key in ("k", "v"):
                    # shape: [1, seq_len, n_heads, head_dim]
                    kv = tensors[kv_key][0, q_len:, :, :]  # [ans_len, n_heads, d_head]
                    norm = kv.norm(dim=-1).mean().item()  # mean over positions & heads
                    layer_norms[kv_key].setdefault(layer, []).append(norm)

        results[label] = {
            kv: {l: float(np.mean(v)) for l, v in norms.items()}
            for kv, norms in layer_norms.items()
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kv_key, title in [
        (ax1, "k", "Key Norm Trajectory"),
        (ax2, "v", "Value Norm Trajectory"),
    ]:
        for label, color in [("truth", "steelblue"), ("hallucinated", "tomato")]:
            ax.plot(
                layers,
                [results[label][kv_key][l] for l in layers],
                label=label.capitalize(),
                marker="o",
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Norm")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model} — Key & Value Norm Trajectory", fontsize=13)
    savefig(fig, outdir, model, "kv_norm_trajectory")
    return results


# ── Experiment 4: Attention entropy ───────────────────────────────────────────


def attention_entropy(df, label_key, outdir, model):
    """
    Experiment 4 — Attention Entropy at Answer Positions.

    Compute per-head Shannon entropy of the attention pattern at answer
    destination positions (post-softmax rows), averaged across heads and
    answer tokens, per layer.

    pattern shape : [1, n_q_heads, seq_dst, seq_src]
    Slice         : pattern[0, :, q_len:, :] → answer rows only
    Entropy       : -sum(p * log(p + eps)) over the source dimension

    Interpretation:
        1. High entropy → attention is spread flat across all source positions;
           the head is not confidently routing to anything specific.
        2. Low entropy → attention is sharp; the head has a clear source.
        3. Uniform attention over many value vectors averages them toward zero —
           this is the direct mechanistic link between high entropy and the
           hidden state collapse observed in earlier experiments.
        4. Hallucinated positions have consistently higher entropy → diffuse
           attention is a primary signal of uncertainty.
        5. Entropy similar across populations → diffusion is not the mechanism;
           fall back to norm collapse as the primary explanation.

    A reference ceiling of log(seq_len) (maximum possible entropy) is plotted
    as a dashed line for calibration.
    """
    print("\n[Exp 4] Attention entropy ...")
    truth_df, halluc_df = split_populations(df, label_key)
    layers = sorted_layers(df)
    eps = 1e-9

    results = {}
    for label, subset in [("truth", truth_df), ("hallucinated", halluc_df)]:
        layer_ent = {}
        for _, row in subset.iterrows():
            q_len = row["metadata"]["question_len"]
            for layer in layers:
                # pattern: [1, n_q_heads, seq_dst, seq_src]
                p = row["layers"][layer]["pattern"][0, :, q_len:, :]
                ent = -(p * (p + eps).log()).sum(dim=-1).mean().item()
                layer_ent.setdefault(layer, []).append(ent)
        results[label] = {l: float(np.mean(v)) for l, v in layer_ent.items()}

    # Reference ceiling: log(seq_len)
    sample_p = df.iloc[0]["layers"][layers[0]]["pattern"]
    seq_len = sample_p.shape[-1]
    max_ent = float(torch.log(torch.tensor(seq_len, dtype=torch.float)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("truth", "steelblue"), ("hallucinated", "tomato")]:
        ax.plot(
            layers,
            [results[label][l] for l in layers],
            label=label.capitalize(),
            marker="o",
            color=color,
        )
    ax.axhline(
        max_ent,
        linestyle="--",
        color="gray",
        alpha=0.5,
        label=f"Max entropy (log {seq_len})",
    )
    ax.set_title(f"{model} — Attention Entropy at Answer Positions")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Entropy (nats)")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, outdir, model, "attention_entropy")
    return results


# ── Experiment 6: Value cancellation ratio ────────────────────────────────────


def value_cancellation_ratio(df, label_key, outdir, model):
    """
    Experiment 6 — Value Cancellation Ratio at Answer Positions.

    Compute the weighted value output per head (pattern @ values), then
    compare the norm of the sum across heads against the sum of individual
    head norms, per layer, at answer positions only.

    Tensor shapes:
        pattern : [1, n_q_heads, seq_dst, seq_src]
        v       : [1, seq_src, n_kv_heads, head_dim]

    For GQA models, KV heads are expanded to match query heads via
    repeat_interleave before the einsum.

    Ratio = ||sum_h(weighted_v_h)|| / (sum_h(||weighted_v_h||) + eps)

    Interpretation:
        1. Ratio near 1 → all heads write in the same direction (constructive
           interference); the residual stream receives a strong, coherent update.
        2. Ratio near 0 → heads write in opposing directions and cancel each
           other out; the net residual update collapses to near zero even though
           each head is individually active.
        3. This is a fundamentally different failure mode from norm collapse —
           individual heads can have healthy key/value norms and sharp attention
           but still produce a collapsed residual if they destructively interfere.
        4. Low ratio on hallucinated positions → destructive interference is an
           active mechanism driving the hidden state collapse.
        5. Similar ratio across populations → cancellation is not the primary
           cause; norm collapse and entropy are more likely explanations.
    """
    print("\n[Exp 6] Value cancellation ratio ...")
    truth_df, halluc_df = split_populations(df, label_key)
    layers = sorted_layers(df)
    n_q_heads = get_n_q_heads(df)
    eps = 1e-9

    results = {}
    for label, subset in [("truth", truth_df), ("hallucinated", halluc_df)]:
        layer_ratios = {}
        for _, row in subset.iterrows():
            q_len = row["metadata"]["question_len"]
            for layer in layers:
                tensors = row["layers"][layer]

                # pattern: [1, n_q_heads, seq_dst, seq_src]
                pattern = tensors["pattern"][0, :, q_len:, :]  # [n_q, ans, src]

                # v: [1, seq_src, n_kv_heads, head_dim] → [n_kv, src, d]
                v = tensors["v"][0].permute(1, 0, 2)
                v = expand_kv_to_q_heads(v, n_q_heads)  # [n_q, src, d]

                # weighted value per head: [n_q, ans, d]
                wv = torch.einsum("hts,hsd->htd", pattern, v)

                norm_of_sum = wv.sum(dim=0).norm(dim=-1)  # [ans]
                sum_of_norms = wv.norm(dim=-1).sum(dim=0)  # [ans]
                ratio = (norm_of_sum / (sum_of_norms + eps)).mean().item()

                layer_ratios.setdefault(layer, []).append(ratio)
        results[label] = {l: float(np.mean(v)) for l, v in layer_ratios.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("truth", "steelblue"), ("hallucinated", "tomato")]:
        ax.plot(
            layers,
            [results[label][l] for l in layers],
            label=label.capitalize(),
            marker="o",
            color=color,
        )
    ax.axhline(
        1.0, linestyle="--", color="green", alpha=0.4, label="Ratio=1 (constructive)"
    )
    ax.axhline(
        0.0, linestyle="--", color="red", alpha=0.4, label="Ratio=0 (full cancel)"
    )
    ax.set_title(f"{model} — Value Cancellation Ratio at Answer Positions")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cancellation Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, outdir, model, "value_cancellation_ratio")
    return results


# ── Experiment 7: Per-head consistency ────────────────────────────────────────


def per_head_consistency(df, label_key, outdir, model):
    """
    Experiment 7 — Per-Head Consistency Analysis.

    For each (layer, head) pair, compute the mean key norm and value norm at
    answer positions, averaged across samples, separately for hallucinated and
    truthful populations. Outputs 2D matrices of shape [n_layers, n_heads].

    For GQA models, value tensors are expanded to match the number of query
    heads before computing per-head norms.

    Plots:
        - Heatmap of truth key/value norms      [n_layers × n_heads]
        - Heatmap of hallucinated key/value norms
        - Difference heatmap (truth − halluc):
            Blue  → truth has higher norm (head writes more on truthful tokens)
            Red   → halluc has higher norm (head writes more on hallucinated tokens)

    Interpretation:
        1. Uniform color across all heads → global representational failure;
           no specific circuit is implicated.
        2. Specific rows (heads) consistently highlighted → those heads are
           implicated across layers; a sparse circuit may be responsible.
        3. Specific columns (layers) highlighted → failure is concentrated at
           that depth; look at what circuit activates there.
        4. Scattered bright spots → noisy signal; head-level analysis is
           uninformative for this model.
        5. Red in late layers (halluc > truth) → overcompensation: the model
           writes louder but more incoherently on uncertain tokens.

    Note: head indices are model-specific and do not transfer across models.
    Use the functional characterisation of implicated heads (attention pattern
    structure, what token types they attend to) for cross-model comparison.

    Returns a dict with truth/halluc matrices and difference matrices for
    downstream use (e.g. automatic selection of most discriminative head).
    """
    print("\n[Exp 7] Per-head consistency ...")
    truth_df, halluc_df = split_populations(df, label_key)
    layers = sorted_layers(df)
    n_q_heads = get_n_q_heads(df)

    def compute_matrices(subset):
        # Derive actual k/v head counts from tensors — independent of n_q_heads
        # k: [1, seq, n_k_heads, d]  — may differ from n_q_heads in GQA models
        # v: [1, seq, n_v_heads, d]  — same as n_k_heads for all standard models
        sample = subset.iloc[0]["layers"][layers[0]]
        n_k_heads = sample["k"].shape[2]
        n_v_heads = sample["v"].shape[2]

        layer_head_k = {l: {h: [] for h in range(n_k_heads)} for l in layers}
        layer_head_v = {l: {h: [] for h in range(n_v_heads)} for l in layers}

        for _, row in subset.iterrows():
            q_len = row["metadata"]["question_len"]
            for layer in layers:
                tensors = row["layers"][layer]
                k = tensors["k"][0, q_len:, :, :]  # [ans, n_k_heads, d]
                v = tensors["v"][0, q_len:, :, :]  # [ans, n_v_heads, d]

                k_norms = k.norm(dim=-1).mean(dim=0)  # [n_k_heads]
                v_norms = v.norm(dim=-1).mean(dim=0)  # [n_v_heads]

                for h in range(n_k_heads):
                    layer_head_k[layer][h].append(k_norms[h].item())
                for h in range(n_v_heads):
                    layer_head_v[layer][h].append(v_norms[h].item())

        k_mat = np.array(
            [[np.mean(layer_head_k[l][h]) for h in range(n_k_heads)] for l in layers]
        )  # [n_layers, n_k_heads]
        v_mat = np.array(
            [[np.mean(layer_head_v[l][h]) for h in range(n_v_heads)] for l in layers]
        )  # [n_layers, n_v_heads]
        return k_mat, v_mat

    truth_k, truth_v = compute_matrices(truth_df)
    halluc_k, halluc_v = compute_matrices(halluc_df)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    layer_labels = [str(l) for l in layers]

    for col, (t_mat, h_mat, title) in enumerate(
        [
            (truth_k, halluc_k, "Key Norm"),
            (truth_v, halluc_v, "Value Norm"),
        ]
    ):
        diff = t_mat - h_mat
        vmax = max(t_mat.max(), h_mat.max())
        dlim = np.abs(diff).max()
        n_heads_this = t_mat.shape[1]  # use actual head count per matrix

        for row_idx, (mat, label, cmap, vmin_, vmax_) in enumerate(
            [
                (t_mat, f"Truth {title}", "viridis", 0, vmax),
                (h_mat, f"Hallucinated {title}", "viridis", 0, vmax),
                (diff, "Difference (Truth−Halluc)", "RdBu", -dlim, dlim),
            ]
        ):
            ax = axes[row_idx, col]
            im = ax.imshow(
                mat.T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin_, vmax=vmax_
            )
            ax.set_title(label)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layer_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(n_heads_this))
            plt.colorbar(im, ax=ax)

    fig.suptitle(
        f"{model} — Per-Head Consistency: Key & Value Norms", fontsize=13, y=1.01
    )
    savefig(fig, outdir, model, "per_head_consistency")

    # Return matrices for downstream use (e.g. picking best head)
    return {
        "truth": {"k": truth_k, "v": truth_v},
        "hallucinated": {"k": halluc_k, "v": halluc_v},
        "diff_k": truth_k - halluc_k,
        "diff_v": truth_v - halluc_v,
        "layers": layers,
    }


def best_discriminative_head(consistency_results, n_late=5):
    """Return the head index most discriminative in the last n_late layers."""
    diff_k = consistency_results["diff_k"]  # [n_layers, n_heads]
    late = diff_k[-n_late:]
    return int(np.argmax(np.abs(late).mean(axis=0)))


# ── Experiment 8: Question vs answer token norms ──────────────────────────────


def question_answer_token_analysis(df, label_key, outdir, model):
    """
    Experiment 8 — Question Token vs. Answer Token Norms.

    Repeat the key and value norm trajectory from Experiments 2 & 3 but
    separately for:
        - Question tokens : k/v[0, :q_len, :, :]
        - Answer tokens   : k/v[0, q_len:, :, :]

    Plots all four combinations (key/value × question/answer) side by side so
    divergence timing is directly comparable across regions.

    Interpretation:
        1. Question norms similar, answer norms lower on hallucinated →
           the model encoded the question correctly but failed during
           generation; failure is in the decoding/answering circuit.
        2. Question norms already lower on hallucinated records →
           failure begins during question encoding; points back to the
           embedding hypothesis (low-norm embeddings for uncertain inputs).
        3. Question norms lower on hallucinated, answer norms similar →
           question encoded weakly but generation partially recovers;
           some downstream circuit compensates.
        4. Both similar across populations →
           neither region shows norm collapse in k/v space; investigate
           attention entropy and cancellation ratio as primary mechanisms.

    The cleanest finding for hallucination research is case 1: it isolates
    the failure to the generation phase and rules out question misrepresentation
    as a confound.
    """
    print("\n[Exp 8] Question vs answer token norms ...")
    truth_df, halluc_df = split_populations(df, label_key)
    layers = sorted_layers(df)

    results = {}
    for label, subset in [("truth", truth_df), ("hallucinated", halluc_df)]:
        region_norms = {"q_k": {}, "q_v": {}, "a_k": {}, "a_v": {}}
        for _, row in subset.iterrows():
            q_len = row["metadata"]["question_len"]
            for layer in layers:
                k = row["layers"][layer]["k"][0]  # [seq, n_heads, d]
                v = row["layers"][layer]["v"][0]  # [seq, n_kv, d]
                for prefix, slc in [
                    ("q", slice(None, q_len)),
                    ("a", slice(q_len, None)),
                ]:
                    k_r = k[slc]
                    v_r = v[slc]
                    region_norms[f"{prefix}_k"].setdefault(layer, []).append(
                        k_r.norm(dim=-1).mean().item()
                    )
                    region_norms[f"{prefix}_v"].setdefault(layer, []).append(
                        v_r.norm(dim=-1).mean().item()
                    )
        results[label] = {
            key: {l: float(np.mean(vs)) for l, vs in norms.items()}
            for key, norms in region_norms.items()
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, key, title in [
        (axes[0, 0], "q_k", "Key Norm — Question Tokens"),
        (axes[0, 1], "a_k", "Key Norm — Answer Tokens"),
        (axes[1, 0], "q_v", "Value Norm — Question Tokens"),
        (axes[1, 1], "a_v", "Value Norm — Answer Tokens"),
    ]:
        for label, color in [("truth", "steelblue"), ("hallucinated", "tomato")]:
            ax.plot(
                layers,
                [results[label][key][l] for l in layers],
                label=label.capitalize(),
                marker="o",
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Norm")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model} — Question vs Answer Token Norms", fontsize=13)
    savefig(fig, outdir, model, "question_answer_token_norms")
    return results


# ── Discriminative head attention visualisation ────────────────────────────────


def head_attention_visualization(df, label_key, head_idx, outdir, model, n_examples=3):
    """
    Discriminative Head Attention Visualisation.

    For the most discriminative head identified in Experiment 7, plot the
    attention pattern at answer destination positions across the late layers
    of the network (last 30% of depth by default), for n_examples from each
    population.

    Tensor slice:
        pattern[0, head_idx, q_len:, :] → [answer_len, seq_src]

    The attention sink at position 0 is zeroed before plotting so that
    non-sink attention is visible. A white vertical line marks the Q|A
    boundary. Each cell is annotated with the fraction of non-sink attention
    directed at question tokens (←Q score).

    head_idx is automatically clamped to [0, n_heads-1] so the function is
    safe to call even if the derived head index exceeds the model's head count.

    Saves two figures:
        {model}_head{head_idx}_attn_truth.png
        {model}_head{head_idx}_attn_hallucinated.png
    """
    print(f"\n[Head viz] Head {head_idx} attention patterns ...")
    truth_df, halluc_df = split_populations(df, label_key)
    all_layers = sorted_layers(df)
    late_layers = late_layer_cutoff(all_layers, fraction=0.7)

    def plot_population(subset, pop_label):
        examples = list(subset.head(n_examples).iterrows())
        n_cols = len(examples)
        n_rows = len(late_layers)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 7, n_rows * 2), sharex="col"
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        color = "steelblue" if pop_label == "Truth" else "tomato"
        fig.suptitle(
            f"Head {head_idx} Attention — {pop_label} "
            f"(Layers {late_layers[0]}–{late_layers[-1]})",
            fontsize=12,
            fontweight="bold",
            color=color,
            y=1.01,
        )

        for col, (_, row) in enumerate(examples):
            q_len = row["metadata"]["question_len"]
            axes[0, col].set_title(
                f"Q: {row['metadata']['question'][:60]}...\n"
                f"A: {row['metadata']['model_answer'][:50]}...",
                fontsize=7,
                pad=5,
            )
            for r, layer in enumerate(late_layers):
                ax = axes[r, col]
                pattern = row["layers"][layer]["pattern"]
                # guard: head_idx must exist in this model
                n_heads = pattern.shape[1]
                h = min(head_idx, n_heads - 1)
                attn = pattern[0, h, q_len:, :].float().numpy()  # [ans, src]

                attn_plot = attn.copy()
                attn_plot[:, 0] = 0  # suppress attention sink
                vmax = attn_plot.max() if attn_plot.max() > 0 else 1.0

                ax.imshow(attn_plot, aspect="auto", cmap="hot", vmin=0, vmax=vmax)
                ax.axvline(x=q_len - 0.5, color="white", linewidth=1.5)

                q_frac = attn[:, 1:q_len].sum() / (attn[:, 1:].sum() + 1e-9)
                ax.set_title(f"←Q: {q_frac:.2f}", fontsize=5, pad=1, color="cyan")
                ax.set_yticks([])
                ax.set_xticks([])
                if col == 0:
                    ax.set_ylabel(f"L{layer}", fontsize=8, rotation=0, labelpad=28)

        plt.tight_layout()
        safe = model.replace("/", "_")
        fname = os.path.join(
            outdir, f"{safe}_head{head_idx}_attn_{pop_label.lower()}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    plot_population(truth_df, "Truth")
    plot_population(halluc_df, "Hallucinated")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, label_key = load_data(args.file, args.label_key)

    key_value_norm_trajectory(df, label_key, args.outdir, args.model)
    attention_entropy(df, label_key, args.outdir, args.model)
    value_cancellation_ratio(df, label_key, args.outdir, args.model)
    consistency = per_head_consistency(df, label_key, args.outdir, args.model)
    question_answer_token_analysis(df, label_key, args.outdir, args.model)

    # Automatically pick the most discriminative head from Exp 7 results
    best_head = best_discriminative_head(consistency)
    print(f"\nMost discriminative head (from Exp 7): {best_head}")
    head_attention_visualization(df, label_key, best_head, args.outdir, args.model)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
