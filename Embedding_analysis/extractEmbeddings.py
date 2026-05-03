from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import re

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
model = HookedTransformer.from_pretrained(MODEL_NAME)

MAX_SAMPLES = 100
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.1

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
            tokens, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
        )

        # Re-run full sequence with cache to get all hooks
        _, cache = model.run_with_cache(generated)

    # Extract before deleting
    hook_embed = cache["hook_embed"].cpu()
    layer_0_input = cache["blocks.0.hook_resid_pre"].cpu()

    del cache
    torch.cuda.empty_cache()

    model_answer = model.to_string(generated[0, tokens.shape[1] :])

    records.append(
        {
            "metadata": {
                "idx": i,
                "question": question,
                "answer": answer,
                "model_answer": model_answer,
                "question_len": tokens.shape[1],
                "total_len": generated.shape[1],
                "answer_len": generated.shape[1] - tokens.shape[1],
            },
            "hook_embed": hook_embed,
            "hook_pos_embed": layer_0_input,
        }
    )

# ── Save ───────────────────────────────────────────────────────────────────────
out_name = f"{re.sub('/', '_', MODEL_NAME)}_embeddings.pt"
torch.save(records, out_name)
print(f"\nSaved {len(records)} records → {out_name}")
