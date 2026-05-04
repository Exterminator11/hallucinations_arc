# H-Neurons — Replication Guide

## Overview

This experiment consists of two parts: ablating H-neurons to extract zeroed response states, then training probes on those states using the same methodology as the Probes experiment.

> **Note:** Ensure `MODEL_NAME` and the `.pkl` file path are updated consistently when testing on a different model.

---

## Pipeline Steps

### Part 1 — H-Neuron Ablation & State Extraction

Run `H_neurons_extractor_ablator.ipynb` to identify H-neurons, zero them out, and save the resulting response states.

**Outputs:**
- `MODEL_NAME_1iter_zeroed_response_states.pkl` — ablated response states
- `MODEL_NAME_1iter_post_ablation.png` — visualization of hidden state geometry after ablation

---

### Part 2 — Probe Training on Ablated States

Pass the `.pkl` file from Part 1 into `probes_H_neurons.ipynb` to train linear probes on the ablated states. This follows the same probing methodology used in the Probes experiment.

**Input:** `MODEL_NAME_1iter_zeroed_response_states.pkl`  
**Output:** Probe training results 

---

## File Reference

| Script | Role | Output |
|---|---|---|
| `H_neurons_extractor_ablator.ipynb` | Ablates H-neurons and extracts zeroed states | `MODEL_NAME_1iter_zeroed_response_states.pkl`, `MODEL_NAME_1iter_post_ablation.png` |
| `probes_H_neurons.ipynb` | Trains probes on ablated states | Probe results |