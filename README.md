# Hallucinations Arc — *Universal Lies*

## Overview

This document describes the experimental components of the **Universal Lies** project, a research effort within the Hallucinations Arc. It outlines which experiments are part of the core study and which are supplementary.

---

## Core Experiments

The following experiments constitute the final experimental pipeline:

1. **Geometric Characterization** — Analysis of the geometric structure of hidden states associated with hallucinated and truthful model outputs.

2. **KV Cache Analysis** — Investigation of attention mechanisms, focusing on identifying the most discriminative attention head and characterizing how attention patterns relate to it.

3. **H-Neurons** — Examination of neuron-level activations implicated in hallucination behavior.

4. **Probes** — Linear classifier probes trained on intermediate representations to assess the discriminability of hallucinated versus truthful states across layers.

---

## Supplementary Experiments

The following experiments are conducted outside the core pipeline and are provided for additional context:

1. **Embedding Analysis** — Exploration of token and representation embeddings as a complementary lens on hallucination-related structure.

---

## Replication Requirements

> **Note:** Replicating these experiments requires a **Google Colab Pro** subscription and an active **Ollama** account.