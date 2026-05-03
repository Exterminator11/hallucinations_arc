import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def load_data(file_path, label_key):
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


def pool_states(states):
    """Mean pool over answer tokens: [1, answer_len, d_model] -> [d_model]"""
    pooled = []
    for s in states:
        arr = s.numpy() if not isinstance(s, np.ndarray) else s
        if arr.ndim == 3:
            arr = arr[0]  # [answer_len, d_model]
        pooled.append(arr.mean(axis=0))  # [d_model]
    return np.array(pooled)  # [N, d_model]


def analyse_embeddings(args, model_name, plot_name, truths, hallucinations):
    tt = pool_states(truths)
    ff = pool_states(hallucinations)

    scaler = StandardScaler()
    tt_scaled = scaler.fit_transform(tt)
    ff_scaled = scaler.transform(ff)

    pca = PCA(n_components=2)
    tt_2d = pca.fit_transform(tt_scaled)
    ff_2d = pca.transform(ff_scaled)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tt_2d[:, 0], tt_2d[:, 1], color="blue", label="Truth", alpha=0.6, s=20)
    ax.scatter(
        ff_2d[:, 0], ff_2d[:, 1], color="red", label="Hallucination", alpha=0.6, s=20
    )
    ax.set_title(plot_name, fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{model_name}_{plot_name}.png"), dpi=150)
    plt.close()


def plot_embeddings_before_and_after_positional_embeddings(args, df):
    truths_hook_embed, hallucinations_hook_embed = [], []
    truths_resid_pre, hallucinations_resid_pre = [], []

    for _, row in df.iterrows():
        q_len = row["metadata"]["question_len"]
        label = row["metadata"]["hallucination_label"]

        hook_embed = row["hook_embed"][:, q_len:, :]
        resid_pre = row["hook_pos_embed"][:, q_len:, :]

        if label == 0:
            truths_hook_embed.append(hook_embed)
            truths_resid_pre.append(resid_pre)
        else:
            hallucinations_hook_embed.append(hook_embed)
            hallucinations_resid_pre.append(resid_pre)

    analyse_embeddings(
        args, args.model, "hook_embed", truths_hook_embed, hallucinations_hook_embed
    )
    analyse_embeddings(
        args,
        args.model,
        "blocks.0.hook_resid_pre",
        truths_resid_pre,
        hallucinations_resid_pre,
    )

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, label_key = load_data(args.file, args.label_key)

    plot_embeddings_before_and_after_positional_embeddings(args, df)


if __name__ == "__main__":
    main()
