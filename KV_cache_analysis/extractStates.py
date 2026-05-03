from transformer_lens import HookedTransformer

MODEL_NAME = "mistral-7b-instruct"
model = HookedTransformer.from_pretrained(MODEL_NAME)
from datasets import load_dataset
import torch
import re

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_SAMPLES = 500
MAX_NEW_TOK = 50
TEMPERATURE = 0.1

layers = model.cfg.n_layers
ALL_LAYERS = list(range(0, layers, 2))

# ── Dataset ────────────────────────────────────────────────────────────────────
truthful_qa = load_dataset(
    "domenicrosati/TruthfulQA", split="train", streaming=True
).take(MAX_SAMPLES)

records = []

for i, data in enumerate(truthful_qa):
    question = data["Question"]
    answer = data["Best Answer"]

    # Tokenize question
    tokens = model.to_tokens(question)

    with torch.no_grad():
        # Generate answer
        generated = model.generate(
            tokens, max_new_tokens=MAX_NEW_TOK, temperature=TEMPERATURE
        )

        # Re-run full sequence with cache to get all hooks
        _, cache = model.run_with_cache(generated)

    # ── Extract per layer ──────────────────────────────────────────────────────
    all_layers = {}

    for layer in ALL_LAYERS:
        keys = cache[f"blocks.{layer}.attn.hook_k"].cpu()
        # shape: [1, seq_len, num_heads, head_dim]

        values = cache[f"blocks.{layer}.attn.hook_v"].cpu()
        # shape: [1, seq_len, num_heads, head_dim]

        pattern = cache[f"blocks.{layer}.attn.hook_pattern"].cpu()
        # shape: [1, num_heads, seq_len, seq_len]
        # post-softmax attention weights — rows sum to 1

        all_layers[layer] = {
            "k": keys,
            "v": values,
            "pattern": pattern,
        }

    # ── Free GPU memory immediately ────────────────────────────────────────────
    del cache
    torch.cuda.empty_cache()

    # ── Decode model answer ────────────────────────────────────────────────────
    model_answer = model.to_string(generated[0, tokens.shape[1] :])

    records.append(
        {
            "metadata": {
                "idx": i,
                "question": question,
                "answer": answer,
                "model_answer": model_answer,
                "question_len": tokens.shape[1],  # boundary: question | answer
                "total_len": generated.shape[1],  # full sequence length
                "answer_len": generated.shape[1] - tokens.shape[1],
            },
            "layers": all_layers,  # dict: {layer_idx: {"k", "v", "pattern"}}
        }
    )

# ── Save ───────────────────────────────────────────────────────────────────────
out_name = f"{re.sub('/', '_', MODEL_NAME)}_all_layers_with_attn.pt"
torch.save(records, out_name)
print(f"\nSaved {len(records)} records → {out_name}")
print(f"Keys per record : metadata, layers")
print(f"Keys per layer  : k, v, pattern")
print(f"Layers saved    : {ALL_LAYERS}")
