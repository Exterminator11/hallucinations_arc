"""
Hallucination Analysis Script
==============================
Runs 5 analyses on attn/mlp streams to find truth vs hallucination signal.

Analyses:
  1. Linear Probe          — is there classifiable signal per layer?
  2. Cosine Similarity     — are truth/halluc directions different?
  3. Temporal Drift        — when during generation does signal diverge?
  4. PCA Visualization     — can you see clusters at the best layer?
  5. Hallucination Direction + AUC — can we make a scalar detector?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.metrics import make_scorer, balanced_accuracy_score

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
data = torch.load(
    f"/Users/rachitdas/Desktop/hallucinations_arc/Qwen Qwen2.5-1.5B-Instruct_streams_labeled.pt"
)

labels = torch.tensor(data["labels"])  # (N,)
truth_idx = (labels == 0).nonzero().squeeze()
halluc_idx = (labels == 1).nonzero().squeeze()
LAYERS = 28
N_TOKENS = 50  # generation timesteps captured

print(
    f"Dataset: {len(labels)} examples | "
    f"Truth: {len(truth_idx)} | Halluc: {len(halluc_idx)}"
)


# ── Helper: mean-pool tokens → (N, 1536) ──────────────────────────────────────
def get_pooled(stream_key, layer):
    return torch.stack(data[stream_key][layer]).mean(dim=1).float()  # (N, 1536)


def get_raw(stream_key, layer):
    return torch.stack(data[stream_key][layer]).float()  # (N, 50, 1536)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT MEANS:
  A logistic regression is trained on the mean-pooled stream vector (1536-dim)
  at each layer to predict truth (0) vs hallucination (1).

  Primary metric is AUC-ROC — immune to class imbalance (14 truth / 80 halluc).
  A dumb classifier that always predicts hallucination gets AUC=0.5, not 0.85.

  - AUC ~0.5  → no separable signal, random
  - AUC  0.6  → weak but real signal
  - AUC  0.7  → meaningful geometric separation
  - AUC  0.8+ → strong signal, worth building on

  Balanced accuracy is reported as a secondary metric — it averages recall
  per class so the 14 truth examples get equal weight to the 80 halluc.

  NOTE: With only 14 truth examples, fold variance will be high.
  Trust layer TRENDS over exact peak layers.
"""
print("\n" + "=" * 60)
print("ANALYSIS 1: LINEAR PROBE")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = labels.numpy()

attn_probe_scores = []  # list of {"auc": float, "bal_acc": float}
mlp_probe_scores = []

for layer in range(LAYERS):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",  # upweights truth examples by ~80/14
                ),
            ),
        ]
    )

    attn_X = get_pooled("attn_streams", layer).numpy()
    mlp_X = get_pooled("mlp_streams", layer).numpy()

    for X, store, name in [
        (attn_X, attn_probe_scores, "Attn"),
        (mlp_X, mlp_probe_scores, "MLP"),
    ]:
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        bal_acc = cross_val_score(
            pipe, X, y, cv=cv, scoring=make_scorer(balanced_accuracy_score)
        ).mean()
        store.append({"auc": auc, "bal_acc": bal_acc})

    print(
        f"  Layer {layer:2d} | "
        f"Attn AUC: {attn_probe_scores[-1]['auc']:.3f} "
        f"bal-acc: {attn_probe_scores[-1]['bal_acc']:.3f} | "
        f"MLP  AUC: {mlp_probe_scores[-1]['auc']:.3f} "
        f"bal-acc: {mlp_probe_scores[-1]['bal_acc']:.3f}"
    )

best_mlp_layer = int(np.argmax([s["auc"] for s in mlp_probe_scores]))
best_attn_layer = int(np.argmax([s["auc"] for s in attn_probe_scores]))

print(
    f"\n  ★ Best MLP layer:  {best_mlp_layer}  "
    f"(AUC={mlp_probe_scores[best_mlp_layer]['auc']:.3f}  "
    f"bal-acc={mlp_probe_scores[best_mlp_layer]['bal_acc']:.3f})"
)
print(
    f"  ★ Best Attn layer: {best_attn_layer} "
    f"(AUC={attn_probe_scores[best_attn_layer]['auc']:.3f}  "
    f"bal-acc={attn_probe_scores[best_attn_layer]['bal_acc']:.3f})"
)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — COSINE SIMILARITY OF MEAN DIRECTIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT MEANS:
  We compute the mean activation vector for truth examples and the mean
  activation vector for hallucination examples, then measure the cosine
  similarity between them.

  - Cosine sim ≈ 1.0  → the two classes point in almost exactly the same
                         direction in activation space. The difference you
                         see in norms is purely about magnitude, not direction.
                         Hard to exploit for a detector.

  - Cosine sim < 0.99 → the classes diverge directionally. Even if norms are
                         similar, the MODEL is computing something different
                         for truth vs halluc. A directional probe will work.

  - Cosine sim drops sharply at a layer → that layer is where the model
                         "splits" its internal representation of true vs false
                         content. Mechanistically very interesting.
"""

print("\n" + "=" * 60)
print("ANALYSIS 2: COSINE SIMILARITY OF MEAN DIRECTIONS")
print("=" * 60)

attn_cos_sims = []
mlp_cos_sims = []
F = torch.nn.functional

for layer in range(LAYERS):
    for key, store in [("attn_streams", attn_cos_sims), ("mlp_streams", mlp_cos_sims)]:
        X = get_pooled(key, layer)
        t_mean = X[truth_idx].mean(dim=0)
        h_mean = X[halluc_idx].mean(dim=0)
        cos = F.cosine_similarity(t_mean.unsqueeze(0), h_mean.unsqueeze(0))
        store.append(cos.item())

    print(
        f"  Layer {layer:2d} | Attn cos: {attn_cos_sims[-1]:.5f} "
        f"| MLP cos: {mlp_cos_sims[-1]:.5f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — TEMPORAL DRIFT OVER GENERATION
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT MEANS:
  Instead of mean-pooling the 50 generation tokens, we look at the norm
  of the stream at EACH timestep separately, averaged across examples.

  This answers: when during generation does the model "commit" to hallucinating?

  - If curves diverge early (token 1–10) → hallucination is decided almost
    immediately, likely driven by the prompt context.

  - If curves diverge late (token 30–50) → the model starts coherently but
    drifts into hallucination as it generates more. This suggests a compounding
    error / "hallucination snowball" effect.

  - If curves never diverge → the temporal structure of the stream doesn't
    differ between classes. The signal is static (in the prompt encoding),
    not dynamic (in the generation).

  We plot this at your 3 most informative layers.
"""

print("\n" + "=" * 60)
print("ANALYSIS 3: TEMPORAL DRIFT")
print("=" * 60)

# Use top 3 MLP layers by probe accuracy
top_layers = sorted(range(LAYERS), key=lambda l: mlp_probe_scores[l], reverse=True)[:3]

temporal_results = {}  # layer → {truth_norm, halluc_norm} each (50,)

for layer in top_layers:
    X = get_raw("mlp_streams", layer)  # (N, 50, 1536)
    norms = X.norm(dim=-1)  # (N, 50)

    truth_norm = norms[truth_idx].mean(dim=0).numpy()  # (50,)
    halluc_norm = norms[halluc_idx].mean(dim=0).numpy()  # (50,)
    temporal_results[layer] = {"truth": truth_norm, "halluc": halluc_norm}

    max_gap = np.max(np.abs(truth_norm - halluc_norm))
    max_gap_token = np.argmax(np.abs(truth_norm - halluc_norm))
    print(f"  Layer {layer:2d} | Max gap: {max_gap:.3f} at token {max_gap_token}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — PCA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT MEANS:
  We project the (N, 1536) MLP stream vectors at the best layer down to 2D
  using PCA and plot truth vs hallucination examples as scatter points.

  - Clear separation into distinct clusters → the classes ARE linearly
    separable, a probe will work well, and the model has a clean internal
    representation of truth vs halluc.

  - Overlapping clouds → the signal is weak or non-linear. You'd need a
    more powerful probe (e.g. MLP classifier) or the signal simply isn't there.

  - One class is more "spread out" → that class has higher internal variance.
    Hallucinations being more spread out would suggest they're less coherent —
    the model is more "uncertain" and activates different features each time.

  PCA components 1 and 2 capture the directions of maximum variance.
  If PC1 alone separates the classes, that means truth vs halluc is the
  DOMINANT source of variation in this layer's activation space — very strong.
"""

print("\n" + "=" * 60)
print("ANALYSIS 4: PCA VISUALIZATION")
print("=" * 60)

pca_results = {}
for layer in [best_mlp_layer] + top_layers[:2]:
    X = get_pooled("mlp_streams", layer).numpy()
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
    var = pca.explained_variance_ratio_
    pca_results[layer] = {"X_2d": X_2d, "var": var}
    print(f"  Layer {layer:2d} | PC1 var: {var[0]:.3f} | PC2 var: {var[1]:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — HALLUCINATION DIRECTION VECTOR + AUC
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT MEANS:
  We compute a single "hallucination direction" vector:
      h_dir = mean(halluc_vectors) - mean(truth_vectors)   [normalized to unit length]

  Then we project every example onto this direction to get a scalar score.
  AUC (Area Under ROC Curve) measures how well this scalar score separates classes.

  - AUC = 0.5  → the direction is useless. Projecting onto it gives random scores.
  - AUC = 0.6  → weak signal. Better than chance but not reliable.
  - AUC > 0.7  → real signal. The difference-in-means vector is a meaningful
                  detector. You could threshold it to flag hallucinations.
  - AUC > 0.8  → strong. This direction could be used for activation steering:
                  subtract it from the residual stream at inference time to
                  reduce hallucinations (like RepE / ITI techniques).

  This is the analysis most directly connected to practical intervention.
  A high AUC here means you have a steering vector, not just a classifier.
"""

print("\n" + "=" * 60)
print("ANALYSIS 5: HALLUCINATION DIRECTION + AUC")
print("=" * 60)

mlp_aucs = []
attn_aucs = []

for layer in range(LAYERS):
    for key, store in [("mlp_streams", mlp_aucs), ("attn_streams", attn_aucs)]:
        X = get_pooled(key, layer)
        t_mean = X[truth_idx].mean(dim=0)
        h_mean = X[halluc_idx].mean(dim=0)

        h_dir = h_mean - t_mean
        h_dir = h_dir / (h_dir.norm() + 1e-8)

        scores = (X @ h_dir).numpy()  # scalar projection per example
        auc = roc_auc_score(y, scores)
        store.append(auc)

    print(
        f"  Layer {layer:2d} | MLP AUC: {mlp_aucs[-1]:.3f} "
        f"| Attn AUC: {attn_aucs[-1]:.3f}"
    )

best_mlp_auc_layer = int(np.argmax(mlp_aucs))
print(
    f"\n  ★ Best MLP AUC:  layer {best_mlp_auc_layer} "
    f"({mlp_aucs[best_mlp_auc_layer]:.3f})"
)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING — All 5 analyses in one figure
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0f0f1a")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

layer_x = list(range(LAYERS))
TRUTH_C = "#4fc3f7"
HALLUC_C = "#ef5350"
MLP_C = "#ab47bc"
ATTN_C = "#26a69a"


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color="#aaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(True, color="#2a2a3e", linewidth=0.5)


# ── Plot 1: Probe accuracy curves ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(
    layer_x, mlp_probe_scores, color=MLP_C, lw=2, label="MLP stream", marker="o", ms=4
)
ax1.plot(
    layer_x,
    attn_probe_scores,
    color=ATTN_C,
    lw=2,
    label="Attn stream",
    marker="s",
    ms=4,
)
ax1.axhline(0.5, color="#555", linestyle="--", lw=1, label="Chance (50%)")
ax1.axvline(
    best_mlp_layer,
    color=MLP_C,
    linestyle=":",
    lw=1.5,
    label=f"Best MLP layer ({best_mlp_layer})",
)
style_ax(
    ax1,
    "Analysis 1 — Linear Probe Accuracy per Layer\n"
    "→ Peak = where the model most linearly separates truth vs halluc",
    "Layer",
    "5-Fold CV Accuracy",
)
ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
ax1.set_ylim(0.4, 1.0)

# ── Plot 2: Cosine similarity ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(layer_x, mlp_cos_sims, color=MLP_C, lw=2, label="MLP", marker="o", ms=3)
ax2.plot(layer_x, attn_cos_sims, color=ATTN_C, lw=2, label="Attn", marker="s", ms=3)
ax2.axhline(1.0, color="#555", linestyle="--", lw=1)
style_ax(
    ax2,
    "Analysis 2 — Cosine Similarity\nof Mean Truth vs Halluc Vectors\n"
    "→ Drop = directional split between classes",
    "Layer",
    "Cosine Similarity",
)
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)

# ── Plot 3: Temporal drift (3 subplots) ───────────────────────────────────────
for i, (layer, res) in enumerate(temporal_results.items()):
    ax = fig.add_subplot(gs[1, i])
    tokens = list(range(N_TOKENS))
    ax.plot(tokens, res["truth"], color=TRUTH_C, lw=2, label="Truth")
    ax.plot(tokens, res["halluc"], color=HALLUC_C, lw=2, label="Halluc")
    ax.fill_between(tokens, res["truth"], res["halluc"], alpha=0.15, color="#fff")
    style_ax(
        ax,
        f"Analysis 3 — Temporal Drift\nMLP Layer {layer}\n"
        "→ When does generation diverge?",
        "Generation Token",
        "Mean Norm",
    )
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)

# ── Plot 4: PCA scatter (best layer) ──────────────────────────────────────────
best_pca_layer = best_mlp_layer
pca_d = pca_results[best_pca_layer]
ax4 = fig.add_subplot(gs[2, :2])
ax4.scatter(
    pca_d["X_2d"][truth_idx, 0],
    pca_d["X_2d"][truth_idx, 1],
    c=TRUTH_C,
    alpha=0.55,
    s=25,
    label="Truth",
    edgecolors="none",
)
ax4.scatter(
    pca_d["X_2d"][halluc_idx, 0],
    pca_d["X_2d"][halluc_idx, 1],
    c=HALLUC_C,
    alpha=0.55,
    s=25,
    label="Halluc",
    edgecolors="none",
)
style_ax(
    ax4,
    f"Analysis 4 — PCA of MLP Stream (Layer {best_pca_layer})\n"
    f"→ Cluster separation = linear probe will work | "
    f"Explained var: PC1={pca_d['var'][0]:.2f}, PC2={pca_d['var'][1]:.2f}",
    "PC1",
    "PC2",
)
ax4.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)

# ── Plot 5: AUC curves ────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
ax5.plot(layer_x, mlp_aucs, color=MLP_C, lw=2, label="MLP", marker="o", ms=3)
ax5.plot(layer_x, attn_aucs, color=ATTN_C, lw=2, label="Attn", marker="s", ms=3)
ax5.axhline(0.5, color="#555", linestyle="--", lw=1, label="Chance")
ax5.axvline(
    best_mlp_auc_layer,
    color=MLP_C,
    linestyle=":",
    lw=1.5,
    label=f"Best layer ({best_mlp_auc_layer})",
)
style_ax(
    ax5,
    "Analysis 5 — Halluc Direction AUC\n" "→ >0.7 = usable steering vector",
    "Layer",
    "AUC",
)
ax5.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
ax5.set_ylim(0.4, 1.0)

fig.suptitle(
    "Hallucination Analysis — Qwen2.5-1.5B-Instruct\n"
    "MLP & Attention Stream Decomposition across 28 Layers",
    color="white",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

plt.savefig(
    "hallucination_analysis.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
print("\nPlot saved to hallucination_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(
    f"  Best probe layer (MLP):     {best_mlp_layer}  "
    f"(acc={mlp_probe_scores[best_mlp_layer]:.3f})"
)
print(
    f"  Best probe layer (Attn):    {best_attn_layer} "
    f"(acc={attn_probe_scores[best_attn_layer]:.3f})"
)
print(
    f"  Best AUC layer (MLP):       {best_mlp_auc_layer}  "
    f"(AUC={mlp_aucs[best_mlp_auc_layer]:.3f})"
)
min_cos_mlp = int(np.argmin(mlp_cos_sims))
print(
    f"  Most directionally split (MLP): layer {min_cos_mlp} "
    f"(cos={mlp_cos_sims[min_cos_mlp]:.5f})"
)

print(
    """
HOW TO READ YOUR RESULTS:
  Probe acc > 0.65 → real signal exists, worth pursuing
  Probe acc > 0.75 → strong signal, probe is a reliable detector
  AUC > 0.70       → difference-in-means vector can steer the model
  Cosine sim drop  → model computes qualitatively different things for truth/halluc
  Temporal diverge early  → hallucination is prompt-driven
  Temporal diverge late   → hallucination snowballs during generation
"""
)
