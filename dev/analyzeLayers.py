import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

MODEL_NAME = "nvidia/OpenReasoning-Nemotron-1.5B"
file = f"{re.sub('/', ' ', MODEL_NAME)}_hallucination_states.pkl"
df = pd.read_pickle(file)

def analyzeLayers():
    state_columns = df.columns[df.columns.str.startswith("state_")]
    labels = df["hallucination_label"].values
    
    # Use lists - don't convert to numpy array yet
    truths = []
    hallucinations = []
    
    for col in state_columns:
        truth_states = df[df["hallucination_label"] == 0][col].values
        hallucination_states = df[df["hallucination_label"] == 1][col].values
        truths.append(truth_states)
        hallucinations.append(hallucination_states)
    
    print(f"Number of layers: {len(state_columns)}")
    
    # Create a grid of subplots
    n_layers = len(state_columns)
    n_cols = 6
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_layers):
        try:
            # Flatten each state for this layer
            tt_flattened = np.array([state.flatten() for state in truths[i]])
            ff_flattened = np.array([state.flatten() for state in hallucinations[i]])
            
            # Apply PCA
            pca = PCA(n_components=2)
            tt_2d = pca.fit_transform(tt_flattened)
            ff_2d = pca.transform(ff_flattened)
            
            # Plot on the i-th subplot
            axes[i].scatter(tt_2d[:, 0], tt_2d[:, 1], color="blue", label="Truth", alpha=0.6, s=20)
            axes[i].scatter(ff_2d[:, 0], ff_2d[:, 1], color="red", label="Hallucination", alpha=0.6, s=20)
            axes[i].set_title(f"Layer {i}", fontsize=10)
            axes[i].legend(fontsize=6)
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error at layer {i}: {e}")
            axes[i].text(0.5, 0.5, f'Layer {i}\nError', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(
        f"{re.sub('/', ' ', MODEL_NAME)}_hallucination_states_layer_analysis.png",
        dpi=150, bbox_inches='tight'
    )
    print(f"Saved plot to {re.sub('/', ' ', MODEL_NAME)}_hallucination_states_layer_analysis.png")
    plt.show()

if __name__ == "__main__":
    analyzeLayers()
