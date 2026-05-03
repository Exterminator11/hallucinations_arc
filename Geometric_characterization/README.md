# Geometric Characterization — Replication Guide

## Overview

This pipeline extracts model hidden states, labels them using an LLM-as-a-Judge framework, and produces per-layer geometric characterization plots.

> **Note:** Ensure the model name and input file are updated consistently across all scripts before running.

---

## Pipeline Steps

### Step 1 — Extract Hidden States

Run `extractStates.py` to extract response-level hidden states without labels.

**Output:** `MODEL_NAME_states.pt` *(or equivalent, as configured)*

---

### Step 2 — Label Data

Feed the file from Step 1 into `llmAsAJudge.py`. This script applies the LLM-as-a-Judge framework, labeling each entry by comparing the model's response against the ground truth.

**Input:** Output from Step 1  
**Output:** Labeled states file

---

### Step 3 — Analyze & Visualize

Pass the labeled file from Step 2 into `analyzeLayers.py` to compute and plot the geometric characterization across layers.

**Input:** Output from Step 2  
**Output:** `MODEL_NAME_hallucination_states_layer_analysis_500.png`

---

## File Reference

| Script | Role | Output |
|---|---|---|
| `extractStates.py` | Extracts unlabeled response hidden states | States file |
| `llmAsAJudge.py` | Labels entries via ground-truth comparison | Labeled states file |
| `analyzeLayers.py` | Generates per-layer geometric plots | `MODEL_NAME_hallucination_states_layer_analysis_500.png` |