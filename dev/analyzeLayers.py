import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
file = f"/Users/rachitdas/Desktop/hallucinations_arc/dev/unsloth Meta-Llama-3.1-8B-Instruct_hallucination_states_last.pkl"
df = pd.read_pickle(file)


def pad_and_flatten(states):
    """Flatten each state and pad to the max length with zeros."""
    flattened = [s.flatten() for s in states]
    max_len = max(f.shape[0] for f in flattened)
    padded = np.array(
        [
            np.pad(f, (0, max_len - f.shape[0]), mode="constant", constant_values=0)
            for f in flattened
        ]
    )
    return padded


def analyzeLayers():
    state_columns = df.columns[df.columns.str.startswith("state_")]
    labels = df["hallucination_label"].values

    truths = []
    hallucinations = []
    for col in state_columns:
        truth_states = df[df["hallucination_label"] == 0][col].values
        hallucination_states = df[df["hallucination_label"] == 1][col].values
        truths.append(truth_states)
        hallucinations.append(hallucination_states)

    print(f"Number of layers: {len(state_columns)}")

    n_layers = len(state_columns)
    n_cols = 6
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_layers):
        try:
            # Pad all states in this layer to the same length before flattening
            all_states = np.concatenate([truths[i], hallucinations[i]])
            all_flattened = [s.flatten() for s in all_states]
            max_len = max(
                f.shape[0] for f in all_flattened
            )  # global max for this layer

            tt_flattened = np.array(
                [
                    np.pad(
                        s.flatten(),
                        (0, max_len - s.flatten().shape[0]),
                        mode="constant",
                        constant_values=0,
                    )
                    for s in truths[i]
                ]
            )
            ff_flattened = np.array(
                [
                    np.pad(
                        s.flatten(),
                        (0, max_len - s.flatten().shape[0]),
                        mode="constant",
                        constant_values=0,
                    )
                    for s in hallucinations[i]
                ]
            )

            scalar = StandardScaler()
            tt_scaled = scalar.fit_transform(tt_flattened)
            ff_scaled = scalar.transform(ff_flattened)

            # PCA (fit on truth only)
            pca = PCA(n_components=2)
            tt_2d = pca.fit_transform(tt_scaled)
            ff_2d = pca.transform(ff_scaled)

            axes[i].scatter(
                tt_2d[:, 0], tt_2d[:, 1], color="blue", label="Truth", alpha=0.6, s=20
            )
            axes[i].scatter(
                ff_2d[:, 0],
                ff_2d[:, 1],
                color="red",
                label="Hallucination",
                alpha=0.6,
                s=20,
            )
            axes[i].set_title(f"Layer {i}", fontsize=10)
            axes[i].legend(fontsize=6)
            axes[i].grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error at layer {i}: {e}")
            axes[i].text(
                0.5,
                0.5,
                f"Layer {i}\nError",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].axis("off")

    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{re.sub('/', ' ', MODEL_NAME)}_prompt_hallucination_states_layer_analysis.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(
        f"Saved plot to {re.sub('/', ' ', MODEL_NAME)}_prompt_hallucination_states_layer_analysis.png"
    )


if __name__ == "__main__":
    analyzeLayers()
