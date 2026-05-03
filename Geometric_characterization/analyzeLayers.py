import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
from scipy import stats


MODEL_NAME = "mistralai Mistral-7B-Instruct-v0.2"
file = f"/content/drive/MyDrive/mistralai Mistral-7B-Instruct-v0.2_hallucination_states_500.pkl"

import shutil

shutil.copy(file, "/content/states.pkl")
df = pd.read_pickle("/content/states.pkl")

def mean_pool(states):
    """
    states: list of arrays, each of shape [n_tokens, hidden_dim] or [hidden_dim]
    Returns: np.array of shape [n_samples, hidden_dim]
    """
    arrays = [np.array(s) for s in states]
    arrays = [s[np.newaxis, :] if s.ndim == 1 else s for s in arrays]
    return np.array([s.mean(axis=0) for s in arrays])  # mean over tokens


def analyzeLayers():
    state_columns = df.columns[df.columns.str.startswith("state_")]
    print(f"Columns: {state_columns}")

    truths, hallucinations = [], []
    for col in state_columns:
        truths.append(df[df["hallucination_label"] == 0][col].values)
        hallucinations.append(df[df["hallucination_label"] == 1][col].values)
    print(len(truths[0]))
    print(len(hallucinations[0]))

    n_layers = len(state_columns)
    print(f"Number of layers: {n_layers}")

    n_cols = 6
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    axes = axes.flatten()

    results = []

    for i in range(n_layers):
        try:
            # Use global max_tokens across truth + hallucination for this layer
            all_states = list(truths[i]) + list(hallucinations[i])
            arrays = [np.array(s) for s in all_states]
            arrays = [s[np.newaxis, :] if s.ndim == 1 else s for s in arrays]
    
            tt_flat = mean_pool(truths[i])  # [n_truth, max_tokens*hidden_dim]
            ff_flat = mean_pool(hallucinations[i])  # [n_hall, max_tokens*hidden_dim]

            # Scale on truth, transform hallucinations
            scaler = StandardScaler()
            tt_scaled = scaler.fit_transform(tt_flat)
            ff_scaled = scaler.transform(ff_flat)

            pca = PCA(n_components=2)
            tt_2d = pca.fit_transform(tt_scaled)
            ff_2d = pca.transform(ff_scaled)

            border_color = "lightgrey"
            for spine in axes[i].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

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

    plt.suptitle(
        f"{MODEL_NAME} | Aggregation: mean",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    plot_path = f"{re.sub('/', ' ', MODEL_NAME)}_hallucination_states_layer_analysis_500_mean.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


results = analyzeLayers()
