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

TARGET_LAYERS = [i for i in range(model.config.num_hidden_layers+1)]


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

    # Extract Hidden State Trajectories
    extracted_layers = {f"layer_{l}": [] for l in TARGET_LAYERS}

    # CRITICAL FIX: Loop behavior
    # outputs.hidden_states is a tuple of tuples.
    # Index 0 = Prompt processing (Shape: Batch, Seq, Dim) -> We SKIP this to keep shapes consistent
    # Index 1+ = Generation steps (Shape: Batch, 1, Dim) -> We KEEP these

    # We iterate starting from 1 to capture only the generated answer tokens
    if len(outputs.hidden_states) > 1:
        for i in range(1, len(outputs.hidden_states)):
            token_step_layers = outputs.hidden_states[i]

            for layer_idx in TARGET_LAYERS:
                # Grab the vector, squeeze to 1D, move to CPU
                # Shape becomes (Hidden_Dim,) e.g. (1536,)
                vector = token_step_layers[layer_idx].squeeze().float().cpu().numpy()
                extracted_layers[f"layer_{layer_idx}"].append(vector)
    else:
        # Edge case: Model generated nothing or only 1 token (rare)
        pass

    # Convert lists to clean Numpy Arrays
    # Final Shape: (Num_Generated_Tokens, Hidden_Dim)
    for key in extracted_layers:
        extracted_layers[key] = np.array(extracted_layers[key])

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

    # Simple prompt
    prompt = f"Question: {q}\nAnswer:"

    gen_ans, states = get_research_data(prompt)

    # Store results
    # We dynamically access keys so we don't crash if TARGET_LAYERS changes
    result_row = {
        "question": q,
        "reference": ref,
        "model_output": gen_ans,
    }

    # Add layers dynamically
    for layer in TARGET_LAYERS:
        key = f"layer_{layer}"
        if key in states:
            result_row[f"state_layer_{layer}"] = states[key]

    dataset_results.append(result_row)

    if i % 5 == 0:
        print(f"Processed {i+1}/100")

# Save
df = pd.DataFrame(dataset_results)
df.to_pickle(f"{re.sub("/"," ",MODEL_NAME)}_hallucination_states.pkl")
print(f"Done! Saved to {re.sub("/"," ",MODEL_NAME)}_hallucination_states.pkl")
