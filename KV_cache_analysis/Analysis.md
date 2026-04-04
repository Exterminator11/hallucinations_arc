# Qwen 3 1.7B 

The attention mechanism is not where hallucination lives in Qwen3 1.7B.
Experiments 1–4 establish that when averaged across heads and positions, every measured attention property — key norms, value norms, routing sharpness, and head alignment — is statistically identical between truthful and hallucinated populations. The model allocates the same "computational energy" to attention regardless of output truthfulness.
Experiment 5 provides the first concrete signal: a late-layer, head-specific divergence concentrated in layers 22–27, with Head 7 being the strongest candidate for a retrieval circuit that differentiates the two populations.
The primary signal is almost certainly in the residual stream. The original hidden state geometric collapse (near-zero hidden states for hallucinated outputs, localized representations for truthful ones) observed in prior work is consistent with everything found here:

Attention is uninvolved → the failure is either in the MLP outputs or in how the residual stream carries information forward
Late-layer divergence in per-head norms → the failure crystallizes in the same layers where the residual stream has been building for 20+ layers

# Head 7 Retrieval Analysis — Qwen3 1.7B
## Experiment: Attention Pattern Visualization at Answer Positions (Layers 22–27)

---

## Motivation

Experiment 5 (Per-Head Consistency) identified **Head 7** as the strongest candidate for a
circuit that differentiates truthful from hallucinated responses in Qwen3 1.7B. It was the
only head with a consistent positive key norm difference (truth > hallucinated) across layers
22–26. This experiment visualizes the raw attention patterns of Head 7 at answer positions
across those layers to determine whether the difference in key norms reflects a meaningful
difference in *what the head is attending to*.

---

## Setup

- **Head index:** 7 (out of 8 KV heads, 16 query heads — GQA with group size 2)
- **Layers visualized:** 22–27 (late layers where Exp 5 divergence was observed)
- **Slice:** `pattern[0, head_idx, q_len:, :]` — answer tokens as destination, all source positions
- **Normalization:** Per-cell (`vmax = attn.max()` per layer × example), with position 0
  masked out to remove the attention sink that dominates raw weights
- **Annotation:** `←Q` score = fraction of non-sink attention directed at question tokens
  (`attn[:, 1:q_len].sum() / attn[:, 1:].sum()`)
- **Populations:** 3 truth examples (blue title) vs. 3 hallucinated examples (red title)

---

## Observations

### Layer 22
- **Truth:** Attention is distributed across both the question region and early answer
  positions simultaneously — the head is doing mixed retrieval and local tracking in
  parallel. Vertical stripes (consistent attention to specific question positions across
  all answer tokens) coexist with diffuse spread.
- **Hallucinated:** Attention is already more diagonal — each answer token predominantly
  attends to the token immediately before it. Question-region engagement is sparser and
  less structured than in truth.

### Layer 24
- **Truth:** The pattern sharpens into **sparse but targeted vertical stripes** at specific
  question positions. The head has identified particular content tokens in the question to
  retrieve from, and maintains that retrieval consistently across all answer tokens.
- **Hallucinated:** The diagonal pattern strengthens considerably. The induction behaviour
  is taking over **3 layers earlier** than in the truth population. Question-region stripes
  are largely absent.

### Layer 26
- **Truth:** Vertical stripes in the question region are still active and clearly visible.
  Retrieval is actively competing with induction — both signals are present. The head has
  not yet committed to pure induction.
- **Hallucinated:** Predominantly diagonal, with question-region attention almost entirely
  gone. **Retrieval has fully collapsed by L26** in hallucinated examples.

### Layer 27
- **Truth:** The diagonal dominates, but the question region retains a residual warm
  signal — some retrieval persists even at the final sampled layer.
- **Hallucinated:** Clean, sharp diagonal. The question region is **completely black** —
  zero retrieval activity. The head has fully committed to previous-token tracking.

---

## Key Finding

Head 7 performs two functions simultaneously in late layers:

1. **Retrieval** — attending to specific question token positions to pull factual content
   into the residual stream
2. **Induction** — attending to the immediately preceding answer token (the diagonal pattern)

In truthful responses, both functions run in parallel through layers 22–27. In hallucinated
responses, **induction takes over 3–4 layers earlier**, fully suppressing retrieval by L26.
By L27, question-region attention is zero in hallucinated examples while truth retains a
residual retrieval signal.

This early collapse means the factual content encoded in the question tokens stops
influencing the residual stream **3–4 layers before the final token distribution is
computed** — a window that is mechanistically sufficient to degrade output factuality.

---

## Interpretation

### What this confirms
The retrieval head hypothesis is **partially confirmed**. Head 7 is not a pure retrieval
head — it transitions to induction behaviour in both populations by L27–L28. However, in
layers 22–26, truthful responses show measurably broader and more sustained question-region
attention. The failure mode in hallucinated examples is not that retrieval never occurs, but
that it **collapses prematurely**.

### What this does not confirm
This experiment establishes a **strong correlate**, not a confirmed causal mechanism. Two
interpretations remain consistent with the data:

- **Head 7 is causal:** the loss of question retrieval in L22–L26 directly degrades the
  factual content available to later layers, producing hallucination
- **Head 7 is symptomatic:** the residual stream already carries a degraded question
  representation by L22 due to upstream failure (MLP or earlier layers), and Head 7's
  retrieval collapse is a downstream consequence of having nothing meaningful to retrieve

### Distinguishing causality
To establish whether Head 7 is causal or symptomatic, **activation patching** is required:
swap Head 7's attention output (or pattern) from a truthful forward pass into a hallucinated
forward pass at layers 22–26, and measure whether the final output token distribution shifts
toward the correct answer. If it does, Head 7 is causally upstream of the failure.

---

## Next Steps

| Experiment | What it tests |
|---|---|
| Activation patching on Head 7 | Causality vs. correlation |
| `hook_resid_post` norm + LDA per layer | Where residual stream representations diverge |
| `hook_mlp_out` norm at answer positions | Whether MLP outputs are the upstream cause |
| Same analysis on Head 5 | Whether the finding generalizes to other late-layer heads |