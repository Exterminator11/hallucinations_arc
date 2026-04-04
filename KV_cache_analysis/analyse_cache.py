import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = "/Users/rachitdas/Desktop/hallucinations_arc/KV_cache_analysis/meta-llama_Llama-3.1-8B-Instruct_hallucination_labels_KV.pt"
MODEL = "llama-3.1-8b-instruct"  # for context in plots, if needed

def load_data():
    print(f"Loading data from {FILE_PATH}...")
    data = torch.load(FILE_PATH)
    return pd.DataFrame(data)

"""
Experiment 1 - Key Norm trajectory
Compute mean key norm at answer positions only, averaged across heads, per layer, comparing hallucinated vs. truthful populations.

1. If both populations start similarly and diverge at a specific layer → that layer is where the failure begins
2. If hallucinated key norms are lower from layer 0 → inherited from embeddings, attention isn't the cause
3. If they're similar throughout → key space doesn't reflect the hidden state collapse you observed, look elsewhere
"""


"""
Experiment 2 - Value Norm Trajectory
Compute mean value norm at answer positions only, averaged across heads, per layer, comparing hallucinated vs. truthful populations.

1. If value norms track key norms closely → keys and values are collapsing together, both are reflecting the same upstream failure
2. If value norms diverge where key norms didn't → values carry additional signal that keys missed, the failure is in what gets written not what gets attended to
3. If value norms are similar across populations even where key norms diverged → the model is attending weakly but still writing meaningful content, attention routing is the problem not the information itself
"""
def key_value_norm_trajectory(df):
    mask = df["metadata"].apply(lambda m: m["hallucination_label"])

    results = {}
    for label, subset_df in [("truth", df[mask == 0]), ("hallucinated", df[mask == 1])]:
        layer_norms = {"k": {}, "v": {}}

        for _, row in subset_df.iterrows():
            q_len = row["metadata"]["question_len"]

            for layer, tensors in row["layers"].items():
                for kv_key in ("k", "v"):
                    kv = tensors[kv_key][
                        0, q_len:, :, :
                    ]  # [answer_len, num_heads, head_dim]
                    mean_norm = kv.norm(dim=-1).mean().item()
                    layer_norms[kv_key].setdefault(layer, []).append(mean_norm)

        results[label] = {
            kv_key: {
                layer: torch.tensor(vals).mean().item() for layer, vals in norms.items()
            }
            for kv_key, norms in layer_norms.items()
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = sorted(results["truth"]["k"].keys())
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

    plt.savefig(f"{MODEL}_kv_norm_trajectory.png", dpi=150)

    return results


"""
Experiment 3 - Attention Entropy at Answer Positions
Compute per-head entropy of the attention pattern at answer destination positions, averaged across heads, per layer, comparing hallucinated vs. truthful populations.
What entropy means here:

1. High entropy → attention is spread flat across all source positions, the head isn't confidently routing to anything specific
2. Low entropy → attention is sharp, the head has a clear source it's pulling from
3. Uniform attention over many value vectors averages them toward zero — this is the direct link between high entropy and the collapse you observed
"""
def attention_entropy(df):
    """
    pattern shape: [1, num_heads, seq_len_dst, seq_len_src]
    We want answer positions as *destination* rows → pattern[0, :, q_len:, :]
    Entropy per head per dst position: -sum(p * log(p + eps))
    Then mean over dst positions and heads → scalar per layer per sample
    """
    mask = df["metadata"].apply(lambda m: m["hallucination_label"])
    eps = 1e-9

    results = {}
    for label, subset_df in [("truth", df[mask == 0]), ("hallucinated", df[mask == 1])]:
        layer_entropies = {}

        for _, row in subset_df.iterrows():
            q_len = row["metadata"]["question_len"]

            for layer, tensors in row["layers"].items():
                pattern = tensors["pattern"][
                    0, :, q_len:, :
                ]  # [num_heads, answer_len, seq_len_src]

                # Entropy per head per answer position
                entropy = -(pattern * (pattern + eps).log()).sum(
                    dim=-1
                )  # [num_heads, answer_len]
                mean_entropy = entropy.mean().item()  # scalar

                layer_entropies.setdefault(layer, []).append(mean_entropy)

        results[label] = {
            layer: torch.tensor(vals).mean().item()
            for layer, vals in layer_entropies.items()
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = sorted(results["truth"].keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, color in [("truth", "steelblue"), ("hallucinated", "tomato")]:
        ax.plot(
            layers,
            [results[label][l] for l in layers],
            label=label.capitalize(),
            marker="o",
            color=color,
        )

    # Annotate max entropy (log(seq_len)) as a reference ceiling
    sample_pattern = df.iloc[0]["layers"][layers[0]]["pattern"]
    seq_len = sample_pattern.shape[-1]
    ax.axhline(
        torch.log(torch.tensor(seq_len, dtype=torch.float)).item(),
        linestyle="--",
        color="gray",
        alpha=0.5,
        label=f"Max entropy (log {seq_len})",
    )

    ax.set_title("Attention Entropy at Answer Positions")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Entropy (nats)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.savefig(f"{MODEL}_attention_entropy.png", dpi=150)

    return results

"""
Experiment 4 - Value Cancellation Ratio
Compute the weighted value output per head (pattern @ values), compare the norm of the sum across heads against the sum of norms, per layer, at answer positions only.What the ratio means:
1. Ratio near 1 → all heads writing in the same direction, constructive interference, strong residual update
2. Ratio near 0 → heads writing in opposing directions, cancelling each other out, net residual update collapses to near zero despite each head individually being active
3. This is a fundamentally different failure mode from Experiments 2-4 — the individual heads can have healthy key and value norms and sharp attention, but still produce a collapsed residual if they cancel
"""

def value_cancellation_ratio(df):
    """
    pattern shape : [1, num_heads, seq_len_dst, seq_len_src]
    v shape       : [1, seq_len_src, num_heads, head_dim]

    For each answer position:
        weighted_v per head = pattern[head, dst, :] @ v[:, head, :]  → [head_dim]
        sum_of_norms        = sum over heads of ||weighted_v_h||
        norm_of_sum         = ||sum over heads of weighted_v_h||
        ratio               = norm_of_sum / (sum_of_norms + eps)
    Then average ratio over answer positions → scalar per layer per sample
    """
    mask = df["metadata"].apply(lambda m: m["hallucination_label"])
    eps = 1e-9

    results = {}
    for label, subset_df in [("truth", df[mask == 0]), ("hallucinated", df[mask == 1])]:
        layer_ratios = {}

        for _, row in subset_df.iterrows():
            q_len = row["metadata"]["question_len"]

            for layer, tensors in row["layers"].items():
                pattern = tensors["pattern"][0, :, q_len:, :]       # [16, answer_len, seq_len_src]
                v = tensors["v"][0].permute(1, 0, 2)                # [8, seq_len_src, head_dim]

                n_q_heads, n_kv_heads = pattern.shape[0], v.shape[0]
                group_size = n_q_heads // n_kv_heads
                v_expanded = v.repeat_interleave(group_size, dim=0)  # [16, seq_len_src, head_dim]

                weighted_v = torch.einsum("hts,hsd->htd", pattern, v_expanded)  # [16, answer_len, head_dim]

                norm_of_sum = weighted_v.sum(dim=0).norm(dim=-1)  # [answer_len]
                sum_of_norms = weighted_v.norm(dim=-1).sum(dim=0)  # [answer_len]

                ratio = (norm_of_sum / (sum_of_norms + eps)).mean().item()
                layer_ratios.setdefault(layer, []).append(ratio)

        results[label] = {
            layer: torch.tensor(vals).mean().item()
            for layer, vals in layer_ratios.items()
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = sorted(results["truth"].keys())
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
        1.0,
        linestyle="--",
        color="green",
        alpha=0.4,
        label="Ratio = 1 (fully constructive)",
    )
    ax.axhline(
        0.0,
        linestyle="--",
        color="red",
        alpha=0.4,
        label="Ratio = 0 (full cancellation)",
    )
    ax.set_title("Value Cancellation Ratio at Answer Positions")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cancellation Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.savefig(f"{MODEL}_value_cancellation_ratio.png", dpi=150)

    return results

"""
Experiment 5: Per-Head Consistency AnalysisFor each head separately, compute mean key norm and value norm at answer positions across layers, comparing hallucinated vs. truthful populations.What you're looking for:
1. Whether the collapse is diffuse - all heads equally weak on hallucinated inputs, suggesting a global representational failure
2. Whether the collapse is sparse — a small subset of heads consistently weaker on hallucinated inputs, suggesting a specific circuit is implicated
3. Whether implicated heads are consistent across layers or shift as depth increases
"""
def per_head_consistency(df):
    """
    For each (layer, head): mean key norm and value norm at answer positions,
    averaged across samples — separately for truth and hallucinated populations.
    Output: heatmaps of shape [num_layers, num_heads] for k and v, both populations.
    """
    mask = df["metadata"].apply(lambda m: m["hallucination_label"])

    results = {}
    for label, subset_df in [("truth", df[mask == 0]), ("hallucinated", df[mask == 1])]:
        # {layer: {head: [per-sample mean norms]}}
        layer_head_k = {}
        layer_head_v = {}

        for _, row in subset_df.iterrows():
            q_len = row["metadata"]["question_len"]

            for layer, tensors in row["layers"].items():
                k = tensors["k"][0, q_len:, :, :]  # [answer_len, num_heads, head_dim]
                v = tensors["v"][0, q_len:, :, :]  # [answer_len, num_heads, head_dim]

                # norm over head_dim, mean over answer positions → [num_heads]
                k_norms = k.norm(dim=-1).mean(dim=0)  # [num_heads]
                v_norms = v.norm(dim=-1).mean(dim=0)  # [num_heads]

                for head in range(k_norms.shape[0]):
                    layer_head_k.setdefault(layer, {}).setdefault(head, []).append(
                        k_norms[head].item()
                    )
                    layer_head_v.setdefault(layer, {}).setdefault(head, []).append(
                        v_norms[head].item()
                    )

        # Average across samples → 2D arrays [num_layers, num_heads]
        layers = sorted(layer_head_k.keys())
        n_heads = max(max(d.keys()) for d in layer_head_k.values()) + 1

        k_matrix = np.array(
            [[np.mean(layer_head_k[l][h]) for h in range(n_heads)] for l in layers]
        )
        v_matrix = np.array(
            [[np.mean(layer_head_v[l][h]) for h in range(n_heads)] for l in layers]
        )

        results[label] = {"k": k_matrix, "v": v_matrix}

    # ── Plot: 4 heatmaps (truth/halluc × k/v) + difference heatmaps ──────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    layers_list = sorted(layer_head_k.keys())

    for col, kv_key, title in [(0, "k", "Key Norm"), (1, "v", "Value Norm")]:
        truth_mat = results["truth"][kv_key]
        halluc_mat = results["hallucinated"][kv_key]
        diff_mat = truth_mat - halluc_mat  # positive → truth has higher norm

        vmax = max(truth_mat.max(), halluc_mat.max())

        im0 = axes[0, col].imshow(
            truth_mat.T,
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=vmax,
            cmap="viridis",
        )
        im1 = axes[1, col].imshow(
            halluc_mat.T,
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=vmax,
            cmap="viridis",
        )
        im2 = axes[2, col].imshow(
            diff_mat.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu",
            vmin=-abs(diff_mat).max(),
            vmax=abs(diff_mat).max(),
        )

        for ax, label in zip(
            [axes[0, col], axes[1, col], axes[2, col]],
            [f"Truth {title}", f"Hallucinated {title}", f"Difference (Truth − Halluc)"],
        ):
            ax.set_title(label)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")
            ax.set_xticks(range(len(layers_list)))
            ax.set_xticklabels(layers_list, fontsize=7)

        plt.colorbar(im0, ax=axes[0, col])
        plt.colorbar(im1, ax=axes[1, col])
        plt.colorbar(im2, ax=axes[2, col])

    plt.suptitle("Per-Head Consistency: Key & Value Norms", fontsize=14, y=1.01)
    
    plt.savefig(f"{MODEL}_per_head_consistency.png", dpi=150, bbox_inches="tight")
    

    return results

"""
Experiment 6 - Question Token vs. Answer Token NormsRepeat the key and value norm analysis from Experiments 2 and 3 but on question tokens instead of answer tokens, using question_len as the upper boundary instead of the lower one.What you're looking for:
1. If question token norms are similar across hallucinated and truthful populations → the model encoded the question fine, the failure is purely in generation
2. If question token norms are already lower on hallucinated records → the failure begins during question encoding, pointing back to the embedding hypothesis from our earlier discussion
3. If question norms diverge at a specific layer → that layer is where the model begins to "lose" the question representation on inputs it will later hallucinate on
"""
def question_answer_token_analysis(df):
    """
    Repeat key/value norm trajectory from Exp 1&2 but separately for:
      - question tokens : [:q_len]
      - answer tokens   : [q_len:]
    Plots side by side so divergence timing is directly comparable.
    """
    mask = df["metadata"].apply(lambda m: m["hallucination_label"])
    eps = 1e-9

    results = {}
    for label, subset_df in [("truth", df[mask == 0]), ("hallucinated", df[mask == 1])]:
        layer_norms = {
            "q_k": {},
            "q_v": {},  # question region
            "a_k": {},
            "a_v": {},  # answer region
        }

        for _, row in subset_df.iterrows():
            q_len = row["metadata"]["question_len"]

            for layer, tensors in row["layers"].items():
                k = tensors["k"][0]  # [seq_len, num_heads, head_dim]
                v = tensors["v"][0]  # [seq_len, num_heads, head_dim]

                for region_key, region_slice in [
                    ("q", slice(None, q_len)),
                    ("a", slice(q_len, None)),
                ]:
                    k_region = k[region_slice]  # [region_len, num_heads, head_dim]
                    v_region = v[region_slice]

                    k_norm = k_region.norm(dim=-1).mean().item()
                    v_norm = v_region.norm(dim=-1).mean().item()

                    layer_norms[f"{region_key}_k"].setdefault(layer, []).append(k_norm)
                    layer_norms[f"{region_key}_v"].setdefault(layer, []).append(v_norm)

        results[label] = {
            key: {layer: np.mean(vals) for layer, vals in norms.items()}
            for key, norms in layer_norms.items()
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = sorted(results["truth"]["q_k"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_cfg = [
        (axes[0, 0], "q_k", "Key Norm — Question Tokens"),
        (axes[0, 1], "a_k", "Key Norm — Answer Tokens"),
        (axes[1, 0], "q_v", "Value Norm — Question Tokens"),
        (axes[1, 1], "a_v", "Value Norm — Answer Tokens"),
    ]

    for ax, key, title in plot_cfg:
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

    plt.suptitle("Question vs. Answer Token Norms by Population", fontsize=14)

    plt.savefig(f"{MODEL}_question_answer_token_norms.png", dpi=150)

    return results


def head7_attention_visualization(df, head_idx=7, n_examples=3, late_layers_only=True):
    mask = df["metadata"].apply(lambda m: m["hallucination_label"])
    truth_df = df[mask == 0].head(n_examples)
    halluc_df = df[mask == 1].head(n_examples)

    all_layers = sorted(df.iloc[0]["layers"].keys())
    # Focus on late layers where divergence was observed in Exp 5
    layers = [l for l in all_layers if l >= 20] if late_layers_only else all_layers
    n_layers = len(layers)

    def plot_population(subset_df, label):
        examples = list(subset_df.iterrows())
        n_cols = len(examples)

        fig, axes = plt.subplots(
            n_layers,
            n_cols,
            figsize=(n_cols * 7, n_layers * 1.8),  # more height per row
            sharex="col",
            sharey=False,  # independent y per cell — answer lengths may differ
        )
        if n_layers == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        color = "steelblue" if label == "Truth" else "tomato"
        fig.suptitle(
            f"Head {head_idx} Attention Pattern — {label} (Layers {layers[0]}–{layers[-1]})",
            fontsize=13,
            fontweight="bold",
            y=1.01,
            color=color,
        )

        for col, (_, row) in enumerate(examples):
            q_len = row["metadata"]["question_len"]
            total_len = row["metadata"]["total_len"]
            question = row["metadata"]["question"][:70]
            model_ans = row["metadata"]["model_answer"][:60]

            axes[0, col].set_title(
                f"Ex {col+1}\nQ: {question}...\nA: {model_ans}...", fontsize=7, pad=6
            )

            for row_idx, layer in enumerate(layers):
                ax = axes[row_idx, col]
                pattern = row["layers"][layer]["pattern"]
                attn = (
                    pattern[0, head_idx, q_len:, :].float().numpy()
                )  # [answer_len, src_len]

                # Mask the attention sink at position 0 before plotting
                attn_masked = attn.copy()
                attn_masked[:, 0] = 0  # zero out sink position

                vmax = attn_masked.max() if attn_masked.max() > 0 else 1.0
                ax.imshow(attn_masked, aspect="auto", cmap="hot", vmin=0, vmax=vmax)
                ax.axvline(x=q_len - 0.5, color="white", linewidth=2.0, linestyle="-")

                attn_to_question = attn[:, 1:q_len].sum() / (attn[:, 1:].sum() + 1e-9)
                ax.set_title(f"←Q: {attn_to_question:.2f}", fontsize=5, pad=1, color="cyan")
                ax.set_yticks([])
                ax.set_xticks([])

                if col == 0:
                    ax.set_ylabel(f"L{layer}", fontsize=8, rotation=0, labelpad=28)

                # Label Q|A boundary only on bottom row
                if row_idx == n_layers - 1:
                    ax.text(q_len, attn.shape[0] * 0.05, "Q|A", color="white",
                            fontsize=6, ha="center", va="bottom")

        # x-axis tick labels only on bottom row
        for col, (_, row) in enumerate(examples):
            total_len = row["metadata"]["total_len"]
            q_len = row["metadata"]["question_len"]
            ticks = list(range(0, total_len, max(1, total_len // 10)))
            axes[-1, col].set_xticks(ticks)
            axes[-1, col].set_xticklabels(
                [f"{i}" + ("|A" if i == q_len else "") for i in ticks],
                fontsize=6,
                rotation=45,
            )
            axes[-1, col].set_xlabel("Source Position", fontsize=7)

        plt.tight_layout()
        fname = f"{MODEL}_head{head_idx}_attn_{label.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        
        print(f"Saved → {fname}")

    plot_population(truth_df, "Truth")
    plot_population(halluc_df, "Hallucinated")


def main():
    df = load_data()
    print("Data loaded successfully. Here's a preview:")
    print(df.head())
    
    # Run the experiments
    key_value_norm_trajectory(df)
    attention_entropy(df)
    value_cancellation_ratio(df)
    per_head_consistency(df)
    question_answer_token_analysis(df)
    head7_attention_visualization(df)


main()
