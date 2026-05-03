# KV Cache Analysis — Replication Guide

## Overview

This pipeline extracts keys, values, and post-softmax attention weights via TransformerLens, labels the data using an LLM-as-a-Judge framework, and analyzes attention head discriminability with respect to question and answer tokens.

> **Note:** This experiment was originally run end-to-end in `KV_cache_analysis.ipynb`, included in the same folder. Ensure `MODEL_NAME` and file paths are updated consistently across all scripts before running.

---

## Pipeline Steps

### Step 1 — Extract Attention & KV States

Run `extractStates.py` to extract keys, values, and post-softmax attention weights across all layers using TransformerLens.

**Output:** `MODEL_NAME_all_layers_with_attn.pt`

---

### Step 2 — Label Data

Pass the file from Step 1 into `LlmAsAJudge.py` to annotate each entry as hallucinated or truthful using the LLM-as-a-Judge framework.

**Input:** `MODEL_NAME_all_layers_with_attn.pt`  
**Output:** `MODEL_NAME_hallucination_labels_KV_500.pt`

---

### Step 3 — Analyze KV Cache

Pass the labeled file from Step 2 into `AnalyzeKVCache.py` to identify the most discriminative attention head and characterize how it attends to question and answer tokens.

**Input:** `MODEL_NAME_hallucination_labels_KV_500.pt`  
**Output:** `MODEL_NAME_results_KV_cache_analysis/` *(folder containing analysis plots and results)*

---

## File Reference

| Script | Role | Output |
|---|---|---|
| `extractStates.py` | Extracts keys, values, and attention weights via TransformerLens | `MODEL_NAME_all_layers_with_attn.pt` |
| `LlmAsAJudge.py` | Labels data via LLM-as-a-Judge | `MODEL_NAME_hallucination_labels_KV_500.pt` |
| `AnalyzeKVCache.py` | Identifies discriminative heads and attention patterns | `MODEL_NAME_results_KV_cache_analysis/` |