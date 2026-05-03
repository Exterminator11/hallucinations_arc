"""
Generic KV-cache hallucination analysis.
Works with any TransformerLens model — GQA or MHA, any depth, any head count.

Usage:
    from analysis_generic import run_analysis, load_data

    df, label_key = load_data("path/to/records.pt")
    results = run_analysis(df, label_key, model="my-model-name", outdir="./results")
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

# ── Data loading ───────────────────────────────────────────────────────────────


def load_data(file_path, label_key="hallucination_label"):
    """
    Load labelled records from a .pt file.

    Parameters
    ----------
    file_path : str
        Path to the labelled .pt records file.
    label_key : str
        Key inside each record's metadata dict that holds the 0/1 hallucination
        label. Defaults to "hallucination_label".

    Returns
    -------
    df : pd.DataFrame
        DataFrame of usable records (parse failures dropped).
    label_key : str
        The label key (passed through for convenience).
    """
    print(f"Loading {file_path} ...")
    records = torch.load(file_path, weights_only=False)
    df = pd.DataFrame(records)

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


def best_discriminative_head(consistency_results, n_late=5):
    """Return the head index most discriminative in the last n_late layers."""
    diff_k = consistency_results["diff_k"]  # [n_layers, n_heads]
    late = diff_k[-n_late:]
    return int(np.argmax(np.abs(late).mean(axis=0)))


# ── Experiment 2 & 3: Key / Value norm trajectory ─────────────────────────────


def key_value_norm_trajectory(df, label_key, outdir, model):
    """
    Experiment 2 & 3 — Key & Value Norm Trajectory.

    Compute mean key and value norm at answer positions only (tokens after
    question_len), averaged across heads, per layer, comparing hallucinated
    vs. truthful populations.

    Parameters
    ----------
    df : pd.DataFrame
    label_key : str
    outdir : str
    model : str

    Returns
    -------
    results : dict
        Nested dict {label: {kv_key: {layer: mean_norm}}}.

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

    Parameters
    ----------
    df : pd.DataFrame
    label_key : str
    outdir : str
    model : str

    Returns
    -------
    results : dict
        {label: {layer: mean_entropy}}.

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

    Parameters
    ----------
    df : pd.DataFrame
    label_key : str
    outdir : str
    model : str

    Returns
    -------
    results : dict
        {label: {layer: mean_ratio}}.

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

    Parameters
    ----------
    df : pd.DataFrame
    label_key : str
    outdir : str
    model : str

    Returns
    -------
    dict with keys:
        "truth"        : {"k": np.array [n_layers, n_heads], "v": ...}
        "hallucinated" : {"k": np.array [n_layers, n_heads], "v": ...}
        "diff_k"       : np.array [n_layers, n_heads]  (truth − halluc)
        "diff_v"       : np.array [n_layers, n_heads]
        "layers"       : list of layer indices

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
    """
    print("\n[Exp 7] Per-head consistency ...")
    truth_df, halluc_df = split_populations(df, label_key)
    layers = sorted_layers(df)
    n_q_heads = get_n_q_heads(df)

    def compute_matrices(subset):
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
        )
        v_mat = np.array(
            [[np.mean(layer_head_v[l][h]) for h in range(n_v_heads)] for l in layers]
        )
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
        n_heads_this = t_mat.shape[1]

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

    return {
        "truth": {"k": truth_k, "v": truth_v},
        "hallucinated": {"k": halluc_k, "v": halluc_v},
        "diff_k": truth_k - halluc_k,
        "diff_v": truth_v - halluc_v,
        "layers": layers,
    }


# ── Experiment 8: Question vs answer token norms ──────────────────────────────


def question_answer_token_analysis(df, label_key, outdir, model):
    """
    Experiment 8 — Question Token vs. Answer Token Norms.

    Parameters
    ----------
    df : pd.DataFrame
    label_key : str
    outdir : str
    model : str

    Returns
    -------
    results : dict
        {label: {"q_k"|"q_v"|"a_k"|"a_v": {layer: mean_norm}}}.

    Interpretation:
        1. Question norms similar, answer norms lower on hallucinated →
           failure is in the decoding/answering circuit.
        2. Question norms already lower on hallucinated records →
           failure begins during question encoding.
        3. Question norms lower on hallucinated, answer norms similar →
           question encoded weakly but generation partially recovers.
        4. Both similar across populations →
           neither region shows norm collapse; investigate entropy and
           cancellation ratio instead.
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

    Parameters
    ----------
    df : pd.DataFrame
    label_key : str
    head_idx : int
        Head index to visualise (clamped to valid range automatically).
    outdir : str
    model : str
    n_examples : int
        Number of examples to plot per population. Default 3.

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


# ── Experiment 9: Attention Fraction Analysis ─────────────────────────────────


import warnings


def attention_fraction_analysis(df, label_key, head_idx, outdir, model):
    """
    Experiment 9 — Attention Fraction Analysis for the Most Discriminative Head.
    (Updated to handle variable sequence lengths safely)
    """
    print(f"\n[Exp 9] Attention fraction analysis (head {head_idx}) ...")
    truth_df, halluc_df = split_populations(df, label_key)
    all_layers = sorted_layers(df)
    late_layers = late_layer_cutoff(all_layers, fraction=0.7)
    eps = 1e-9

    def collect(subset):
        mean_fracs = {l: [] for l in late_layers}
        q_fracs = {l: [] for l in late_layers}

        for _, row in subset.iterrows():
            q_len = row["metadata"]["question_len"]
            for layer in late_layers:
                pattern = row["layers"][layer]["pattern"]
                n_heads = pattern.shape[1]
                h = min(head_idx, n_heads - 1)

                attn = pattern[0, h, q_len:, :].float().numpy()

                attn_no_sink = attn.copy()
                attn_no_sink[:, 0] = 0.0

                row_sums = attn_no_sink.sum(axis=-1, keepdims=True) + eps
                frac = attn_no_sink / row_sums
                mean_frac = frac.mean(axis=0)
                mean_fracs[layer].append(mean_frac)

                q_frac = mean_frac[1:q_len].sum()
                q_fracs[layer].append(float(q_frac))

        return mean_fracs, q_fracs

    truth_fracs, truth_q = collect(truth_df)
    halluc_fracs, halluc_q = collect(halluc_df)

    # --- Helper to safely mean variable-length sequences ---
    def pad_and_mean(fracs_list):
        if not fracs_list:
            return np.array([])
        max_len = max(len(f) for f in fracs_list)
        # Pad shorter sequences with NaNs so we can stack them
        padded = np.array(
            [
                np.pad(f, (0, max_len - len(f)), constant_values=np.nan)
                for f in fracs_list
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(padded, axis=0)

    mean_frac = {"truth": {}, "hallucinated": {}}
    fold_change = {"truth": {}, "hallucinated": {}}

    for label, fracs in [("truth", truth_fracs), ("hallucinated", halluc_fracs)]:
        for layer in late_layers:
            mf = pad_and_mean(fracs[layer])
            mean_frac[label][layer] = mf

            # Recalculate uniform baseline dynamically based on actual length
            current_seq_len = len(mf)
            uniform_baseline = 1.0 / max(current_seq_len - 1, 1)
            fold_change[label][layer] = mf / uniform_baseline

    stats = {}
    for layer in late_layers:
        t_vals = truth_q[layer]
        h_vals = halluc_q[layer]
        if not t_vals or not h_vals:
            continue
        u_stat, p_val = mannwhitneyu(t_vals, h_vals, alternative="two-sided")
        fc_q = (np.mean(t_vals) + eps) / (np.mean(h_vals) + eps)
        stats[layer] = {
            "u": float(u_stat),
            "p": float(p_val),
            "fold_change_q": float(fc_q),
        }
        print(
            f"  Layer {layer:2d} | MWU p={p_val:.4f} | "
            f"Q-frac fold-change (truth/halluc) = {fc_q:.3f}"
        )

    # Plot 1: Per-position fold-change heatmap
    print("  Plotting fold-change heatmaps ...")
    fig, axes = plt.subplots(1, 2, figsize=(18, max(4, len(late_layers) * 0.5 + 2)))

    for ax, label, color in [
        (axes[0], "truth", "steelblue"),
        (axes[1], "hallucinated", "tomato"),
    ]:
        fc_mat = np.stack([fold_change[label][l] for l in late_layers])
        vmax = max(2.0, float(np.nanpercentile(fc_mat, 99)))

        # Mask NaNs so they don't plot as weird artifacts
        masked_fc_mat = np.ma.masked_invalid(fc_mat)
        im = ax.imshow(
            masked_fc_mat,
            aspect="auto",
            origin="upper",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=vmax,
        )

        current_seq_len = fc_mat.shape[1]
        ax.axvline(
            x=current_seq_len // 2 - 0.5, color="white", linewidth=1.5, alpha=0.7
        )
        ax.set_title(f"{label.capitalize()} — Fold-change vs. Uniform", color=color)
        ax.set_xlabel("Source token position")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(late_layers)))
        ax.set_yticklabels([str(l) for l in late_layers], fontsize=8)
        plt.colorbar(im, ax=ax, label="Fold-change (1.0 = uniform)")

    fig.suptitle(
        f"{model} — Head {head_idx} Attention Fold-Change vs. Uniform Baseline",
        fontsize=12,
    )
    savefig(fig, outdir, model, f"head{head_idx}_attn_fold_change")

    # Plot 2: Q-region vs A-region mean fraction across late layers
    print("  Plotting Q/A region fraction lines ...")
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, color, q_fracs_d, a_fracs_d in [
        ("truth", "steelblue", truth_q, truth_fracs),
        ("hallucinated", "tomato", halluc_q, halluc_fracs),
    ]:
        q_means = [np.mean(q_fracs_d[l]) for l in late_layers]

        a_means = []
        for l in late_layers:
            mf = mean_frac[label][l]  # Use pre-computed padded means
            mid_idx = len(mf) // 2
            a_sum = np.nansum(mf[mid_idx:])
            total_sum = np.nansum(mf[1:])
            a_means.append(float(a_sum) / max(float(total_sum), eps))

        ax.plot(
            late_layers,
            q_means,
            marker="o",
            color=color,
            label=f"{label.capitalize()} — Q-region",
        )
        ax.plot(
            late_layers,
            a_means,
            marker="s",
            color=color,
            linestyle="--",
            label=f"{label.capitalize()} — A-region",
            alpha=0.6,
        )

    ax.set_title(f"{model} — Head {head_idx}: Q-region vs A-region Attention Fraction")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Fraction (non-sink)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, outdir, model, f"head{head_idx}_qa_region_fraction")

    # Plot 3: Per-sample Q-fraction distribution with stats annotation
    print("  Plotting per-sample Q-fraction distributions ...")
    n_layers = len(late_layers)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(max(12, n_layers * 2.5), 5), sharey=True
    )
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, late_layers):
        if layer not in stats:
            continue
        t_vals = truth_q[layer]
        h_vals = halluc_q[layer]
        s = stats[layer]

        parts = ax.violinplot(
            [t_vals, h_vals],
            positions=[0, 1],
            showmedians=True,
            showextrema=True,
        )
        for pc, color in zip(parts["bodies"], ["steelblue", "tomato"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for part_key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part_key].set_color("black")
            parts[part_key].set_linewidth(1.0)

        rng = np.random.default_rng(42)
        for vals, x_center, color in [(t_vals, 0, "steelblue"), (h_vals, 1, "tomato")]:
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(x_center + jitter, vals, s=12, color=color, alpha=0.5, zorder=3)

        p = s["p"]
        fc = s["fold_change_q"]
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.set_title(f"L{layer}\n{p_str}\nFC={fc:.2f}×", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Truth", "Halluc"], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Q-region attention fraction (non-sink)")
    fig.suptitle(
        f"{model} — Head {head_idx}: Per-Sample Q-Region Attention Fraction\n"
        f"(Mann-Whitney U; FC = truth/halluc fold-change)",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, outdir, model, f"head{head_idx}_q_fraction_distributions")

    return {
        "layers": late_layers,
        "mean_frac": mean_frac,
        "fold_change": fold_change,
        "q_frac_samples": {"truth": truth_q, "hallucinated": halluc_q},
        "stats": stats,
    }


# ── Top-level orchestration ────────────────────────────────────────────────────


def run_analysis(df, label_key, model, outdir=".", label_key_override=None):
    """
    Run all hallucination analysis experiments on a loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Records as returned by load_data().
    label_key : str
        Label key as returned by load_data().
    model : str
        Model name used for output filenames and plot titles.
    outdir : str
        Directory to save plots. Created if it does not exist.

    Returns
    -------
    dict with keys:
        "kv_norms"           : results from key_value_norm_trajectory()
        "entropy"            : results from attention_entropy()
        "cancellation"       : results from value_cancellation_ratio()
        "head_consistency"   : results from per_head_consistency()
        "qa_token_norms"     : results from question_answer_token_analysis()
        "best_head"          : int, most discriminative head index
        "attention_fractions": results from attention_fraction_analysis()
    """
    os.makedirs(outdir, exist_ok=True)

    kv_norms = key_value_norm_trajectory(df, label_key, outdir, model)
    entropy = attention_entropy(df, label_key, outdir, model)
    cancellation = value_cancellation_ratio(df, label_key, outdir, model)
    consistency = per_head_consistency(df, label_key, outdir, model)
    qa_norms = question_answer_token_analysis(df, label_key, outdir, model)

    best_head = best_discriminative_head(consistency)
    print(f"\nMost discriminative head (from Exp 7): {best_head}")

    head_attention_visualization(df, label_key, best_head, outdir, model)
    frac_results = attention_fraction_analysis(df, label_key, best_head, outdir, model)

    print("\nAll experiments complete.")

    return {
        "kv_norms": kv_norms,
        "entropy": entropy,
        "cancellation": cancellation,
        "head_consistency": consistency,
        "qa_token_norms": qa_norms,
        "best_head": best_head,
        "attention_fractions": frac_results,
    }

FILE = "/content/drive/MyDrive/qwen3-1.7b_hallucination_labels_KV_500.pt"
MODEL = "qwen3-1.7b"

df, label_key = load_data(FILE)
results = run_analysis(
    df,
    "hallucination_label",
    model=MODEL,
    outdir=f"./{MODEL}_results_KV_cache_analysis",
)
