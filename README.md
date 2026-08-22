# Hallucinations Arc — *Universal Lies*

## Overview

This repository contains the experiments for the **Universal Lies** project. Together, they study hallucinations in language models at the representation, attention, and neuron levels, and evaluate how well hallucinated and truthful responses can be separated.

**Paper:** [`Universal Lies`](Universal_lies.pdf)

## Experiments

### 1. Geometric Characterization

Analyzes the geometry of hidden states associated with hallucinated and truthful responses across model layers. The pipeline extracts response states, labels them with an LLM-as-a-Judge, and generates per-layer visualizations.

- Guide: [`Geometric_characterization/README.md`](Geometric_characterization/README.md)
- Main notebook: [`whatDoHallucinationsLookLike.ipynb`](Geometric_characterization/whatDoHallucinationsLookLike.ipynb)
- Pipeline: [`extractStates.py`](Geometric_characterization/extractStates.py) → [`LlmAsAJudge.py`](Geometric_characterization/LlmAsAJudge.py) → [`analyzeLayers.py`](Geometric_characterization/analyzeLayers.py)
- Included analyses: Qwen 2.5 1.5B, Qwen 3 1.7B, Qwen 3 4B, Llama 3.1 8B, and Mistral 7B.

### 2. KV Cache Analysis

Examines keys, values, and post-softmax attention weights using TransformerLens. It identifies discriminative attention heads and analyzes their attention to question and answer tokens, including entropy, norm trajectories, fold changes, and value cancellation.

- Guide: [`KV_cache_analysis/README.md`](KV_cache_analysis/README.md)
- Main notebook: [`KV_cache_analysis.ipynb`](KV_cache_analysis/KV_cache_analysis.ipynb)
- Pipeline: [`extractStates.py`](KV_cache_analysis/extractStates.py) → [`LlmAsAJudge.py`](KV_cache_analysis/LlmAsAJudge.py) → [`AnalyzeKVCache.py`](KV_cache_analysis/AnalyzeKVCache.py)
- Included analyses: Qwen 2.5 1.5B, Qwen 3 1.7B, Qwen 3 4B, and Llama 3.1 8B.

### 3. H-Neurons

Investigates neuron-level contributions to hallucination behavior. H-neurons are ablated, zeroed response states are extracted, and probes are trained on the ablated representations.

- Guide: [`H-neurons/README.md`](H-neurons/README.md)
- Ablation notebook: [`H_neuron_extractor_ablator.ipynb`](H-neurons/H_neuron_extractor_ablator.ipynb)
- Probe notebook: [`probes_H_neurons.ipynb`](H-neurons/probes_H_neurons.ipynb)
- Included ablation results: Qwen 2.5 1.5B, Qwen 3 1.7B, Qwen 3 4B, and Llama 3.1 8B.

### 4. Probes

Trains linear probes on intermediate hidden states to measure the layer-wise separability of hallucinated and truthful responses. This experiment consumes labeled states from Geometric Characterization.

- Guide: [`Probes/README.md`](Probes/README.md)
- Notebook: [`Probes.ipynb`](Probes/Probes.ipynb)
- Included probe results: Qwen 2.5 1.5B, Qwen 3 1.7B, Qwen 3 4B, Llama 3.1 8B, and Mistral 7B.

### 5. Embedding Analysis

Provides a complementary analysis of token and representation embeddings before and after positional embeddings are applied. Embeddings are extracted, labeled with an LLM-as-a-Judge, and visualized.

- Guide: [`Embedding_analysis/README.md`](Embedding_analysis/README.md)
- Notebook: [`embed_analysis.ipynb`](Embedding_analysis/embed_analysis.ipynb)
- Pipeline: [`extractEmbeddings.py`](Embedding_analysis/extractEmbeddings.py) → [`llmAsAJudgePt.py`](Embedding_analysis/llmAsAJudgePt.py) → [`analyzeEmbeddings.py`](Embedding_analysis/analyzeEmbeddings.py)
- Included embedding visualizations: Qwen 2.5 1.5B, Qwen 3 1.7B, Qwen 3 4B, and Llama 3.1 8B.

## Experiment Relationships

Geometric Characterization produces the labeled hidden-state data used by Probes. H-Neurons follows a similar probing procedure after neuron ablation. KV Cache Analysis and Embedding Analysis are independent complementary analyses of attention/KV states and embedding structure, respectively.

---

## Replication Requirements

> **Note:** Replicating these experiments requires a **Google Colab Pro** subscription and an active **Ollama** account. Update model names and input paths in the scripts or notebooks before running them.
