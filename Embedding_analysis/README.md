# Embedding Analysis — Replication Guide

## Overview

This document describes the steps required to replicate the Embedding Analysis experiment. The pipeline extracts token embeddings, labels them using an LLM-as-a-Judge framework, and produces visualizations before and after positional embeddings are applied.

> **Note:** Ensure the model name and input file are updated consistently across all scripts before running.

---

## Pipeline Steps

### Step 1 — Extract Embeddings

Run `extractEmbeddings.py` to extract hidden states before and after positional embeddings are added.

**Output:** `MODEL_NAME_embeddings.pt`

---

### Step 2 — Label Data

Pass the embeddings file from Step 1 into `LlmAsAJudge.py`. This script applies the LLM-as-a-Judge framework to annotate each sample as hallucinated or truthful.

**Input:** `MODEL_NAME_embeddings.pt`  
**Output:** `MODEL_NAME_hallucination_labels_embeddings.pt`

---

### Step 3 — Analyze & Visualize

Pass the labeled data from Step 2 into `analyzeEmbeddings.py` to generate visualizations comparing embedding structure before and after positional encoding.

**Input:** `MODEL_NAME_hallucination_labels_embeddings.pt`  
**Output:** `MODEL_NAME_PLOT_NAME.png` (one file per plot)

---

## File Reference

| Script | Role | Output |
|---|---|---|
| `extractEmbeddings.py` | Extracts pre- and post-positional embeddings | `MODEL_NAME_embeddings.pt` |
| `LlmAsAJudge.py` | Labels data via LLM-as-a-Judge | `MODEL_NAME_hallucination_labels_embeddings.pt` |
| `analyzeEmbeddings.py` | Generates embedding visualizations | `MODEL_NAME_PLOT_NAME.png` |