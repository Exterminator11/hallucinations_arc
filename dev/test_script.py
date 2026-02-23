import torch
import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    output_hidden_states=True,
).to(device)


# ----------------------------------------------------------------
# Step 1: Get k responses with hidden states
# ----------------------------------------------------------------
def get_k_responses_with_hidden_states(prompt, k=5, max_new_tokens=50):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    responses = []
    all_hidden_states = []

    for i in range(k):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0][inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response_text)

        # output.hidden_states is a tuple over generated steps
        # Each step: tuple of (n_layers+1) tensors, each (1, 1, hidden_dim)
        # We keep all steps so we can mean-pool over generated tokens
        all_hidden_states.append(output.hidden_states)

        print(f"[{i+1}/{k}] {response_text}")

    return responses, all_hidden_states


# ----------------------------------------------------------------
# Step 2: Extract trajectories — mean-pool over ALL generated tokens
#         This captures the full response, not just one token
# ----------------------------------------------------------------
def extract_trajectories(all_hidden_states):
    trajectories = []
    for hs_steps in all_hidden_states:
        # hs_steps: tuple of steps, each step is tuple of (n_layers+1) x (1, 1, hidden)
        n_layers = len(hs_steps[0])
        n_steps = len(hs_steps)

        # Stack all generated tokens per layer, then mean pool
        # result: (n_layers, hidden_dim)
        layer_means = []
        for layer in range(n_layers):
            # (n_steps, hidden_dim)
            layer_tokens = np.stack(
                [
                    hs_steps[step][layer][0, 0, :].cpu().float().numpy()
                    for step in range(n_steps)
                ]
            )
            layer_means.append(layer_tokens.mean(axis=0))

        trajectories.append(np.stack(layer_means))  # (n_layers, hidden_dim)

    return np.stack(trajectories)  # (k, n_layers, hidden_dim)


# ----------------------------------------------------------------
# Step 3: Plot trajectories across layers in a single shared PCA space
# ----------------------------------------------------------------
def plot_trajectories(
    trajectories, responses, prompt, save_path="trajectories_across_layers.png"
):
    k, n_layers, hidden_dim = trajectories.shape

    flat = trajectories.reshape(k * n_layers, hidden_dim)
    pca = PCA(n_components=2)
    flat_2d = pca.fit_transform(flat)
    traj_2d = flat_2d.reshape(k, n_layers, 2)

    var_explained = pca.explained_variance_ratio_[:2].sum()

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    cmap = matplotlib.colormaps.get_cmap("tab10")

    for i in range(k):
        color = cmap(i / max(k - 1, 1))
        xs, ys = traj_2d[i, :, 0], traj_2d[i, :, 1]

        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1.5, zorder=2)

        for layer in range(n_layers):
            alpha = 0.2 + 0.8 * (layer / max(n_layers - 1, 1))
            ax.scatter(xs[layer], ys[layer], color=color, s=30, alpha=alpha, zorder=3)

        ax.scatter(
            xs[0],
            ys[0],
            color=color,
            s=120,
            marker="o",
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.scatter(
            xs[-1],
            ys[-1],
            color=color,
            s=180,
            marker="*",
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )

        x_range = flat_2d[:, 0].max() - flat_2d[:, 0].min()
        y_range = flat_2d[:, 1].max() - flat_2d[:, 1].min()
        ax.text(
            xs[-1] + (x_range * 0.01),
            ys[-1] + (y_range * 0.01),
            f"R{i+1}",
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

    reference_layers = sorted(
        set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1])
    )
    for layer in reference_layers:
        ax.annotate(
            f"L{layer}",
            (traj_2d[0, layer, 0], traj_2d[0, layer, 1]),
            fontsize=7,
            color="#aaaaaa",
            xytext=(4, 4),
            textcoords="offset points",
        )

    legend_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            color=cmap(i / max(k - 1, 1)),
            linewidth=2,
            label=f"R{i+1}: {responses[i][:60]}{'...' if len(responses[i]) > 60 else ''}",
        )
        for i in range(k)
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7,
        facecolor="#1a1d24",
        edgecolor="#444",
        labelcolor="white",
        framealpha=0.85,
        title="Responses",
        title_fontsize=8,
    )
    leg.get_title().set_color("#aaaaaa")

    marker_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markersize=8,
            label="Layer 0 (start)",
            linestyle="None",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="white",
            markersize=10,
            label=f"Layer {n_layers-1} (end)",
            linestyle="None",
        ),
    ]
    ax.legend(
        handles=marker_handles,
        loc="lower left",
        fontsize=7,
        facecolor="#1a1d24",
        edgecolor="#444",
        labelcolor="white",
        framealpha=0.85,
    )
    ax.add_artist(leg)

    ax.set_title(
        f"Hidden State Trajectories Across {n_layers} Layers — {k} Responses\n"
        f'"{prompt}"\n'
        f"Shared PCA (mean-pooled generated tokens)  |  variance explained: {var_explained:.1%}  |  ○ = Layer 0   ★ = Layer {n_layers-1}",
        fontsize=9,
        color="white",
        pad=12,
    )
    ax.set_xlabel("PC1", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("PC2", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved to {save_path}")

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    prompt = "What is the currently accepted explanation for the 'Little Red Dots' discovered by the James Webb Space Telescope, and has the 'black hole star' theory been officially proven or debunked as of February 2023?"  # <-- Change this prompt to test different questions!
    k = 10

    responses, hidden_states = get_k_responses_with_hidden_states(prompt, k=k)
    trajectories = extract_trajectories(hidden_states)
    print(f"\nTrajectories shape: {trajectories.shape}")  # (k, n_layers, hidden_dim)

    plot_trajectories(trajectories, responses, prompt)
