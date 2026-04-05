import torch
import gc
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

# 1. Load Dataset
truthful_qa_dataset = load_dataset(
    "domenicrosati/TruthfulQA", split="train", streaming=True
).take(100)

# 2. Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 3. Load Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE
)

TARGET_LAYERS = [i for i in range(model.config.num_hidden_layers + 1)]


def get_prompt_states(question):
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    prompt_layers = {}
    for layer_idx in TARGET_LAYERS:
        # Grab only the last token: shape (hidden_dim,)
        last_token = (
            outputs.hidden_states[layer_idx].squeeze(0)[-1].float().cpu().numpy()
        )
        prompt_layers[f"layer_{layer_idx}"] = last_token

    del outputs, inputs
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return prompt_layers


def get_model_answer(question):
    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs.input_ids.shape[-1]
    gen_tokens = outputs[0][prompt_len:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    del outputs, inputs
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return answer


# --- Main Loop ---
dataset_results = []

for i, data in enumerate(truthful_qa_dataset):
    q = data["Question"]
    ref = data["Best Answer"]

    prompt = f"Question: {q}\nAnswer:"

    model_answer = get_model_answer(prompt)
    prompt_states = get_prompt_states(prompt)

    result_row = {
        "question": q,
        "reference": ref,
        "model_output": model_answer,
    }

    # Each value is now shape (hidden_dim,) — last token only
    for layer_idx in TARGET_LAYERS:
        key = f"layer_{layer_idx}"
        result_row[f"state_{key}"] = prompt_states[key]

    dataset_results.append(result_row)

    if i % 5 == 0:
        print(f"Processed {i+1}/100")

# Save
df = pd.DataFrame(dataset_results)
save_name = f"{re.sub('/', '_', MODEL_NAME)}_prompt_without_labels_last.pkl"
df.to_pickle(save_name)
print(f"Done! Saved to {save_name}")
