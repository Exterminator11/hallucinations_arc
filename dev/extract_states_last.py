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


def get_research_data(question):
    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_hidden_states=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract Answer Text
    prompt_len = inputs.input_ids.shape[-1]
    gen_tokens = outputs.sequences[0][prompt_len:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Extract Hidden States for Last Generated Token Only
    extracted_layers = {f"layer_{l}": None for l in TARGET_LAYERS}

    if len(outputs.hidden_states) > 1:
        # outputs.hidden_states[-1] = last generation step
        # Each entry is a tuple of (num_layers+1) tensors, each shape: (batch, 1, hidden_dim)
        last_token_layers = outputs.hidden_states[-1]

        for layer_idx in TARGET_LAYERS:
            # Squeeze to 1D: (hidden_dim,) e.g. (1536,)
            vector = last_token_layers[layer_idx].squeeze().float().cpu().numpy()
            extracted_layers[f"layer_{layer_idx}"] = vector
    else:
        # Edge case: model generated nothing
        for layer_idx in TARGET_LAYERS:
            extracted_layers[f"layer_{layer_idx}"] = np.zeros(model.config.hidden_size)

    # Cleanup
    del outputs, inputs
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return answer, extracted_layers


# --- Main Loop ---
dataset_results = []

for i, data in enumerate(truthful_qa_dataset):
    q = data["Question"]
    ref = data["Best Answer"]

    prompt = f"Question: {q}\nAnswer:"
    gen_ans, states = get_research_data(prompt)

    result_row = {
        "question": q,
        "reference": ref,
        "model_output": gen_ans,
    }

    for layer in TARGET_LAYERS:
        key = f"layer_{layer}"
        if states[key] is not None:
            result_row[f"state_layer_{layer}"] = states[key]

    dataset_results.append(result_row)

    if i % 5 == 0:
        print(f"Processed {i+1}/100")

# Save
df = pd.DataFrame(dataset_results)
df.to_pickle(f"{re.sub('/', ' ', MODEL_NAME)}_without_labels_last.pkl")
print(f"Done! Saved to {re.sub('/', ' ', MODEL_NAME)}_without_labels_last.pkl")
