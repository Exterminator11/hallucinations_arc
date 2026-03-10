import transformer_lens
from collections import OrderedDict
import torch
import json

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformer_lens.HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype="bfloat16")
LAYERS = model.cfg.n_layers


def plot_residual_streams(prompt):
    # 1. Initialize lists BEFORE the loop, and OUTSIDE token re-init
    attn_dict = {layer: [] for layer in range(LAYERS)}
    mlp_dict = {layer: [] for layer in range(LAYERS)}

    # 2. Tokenize ONCE before the loop
    tokens = model.to_tokens(prompt)

    for _ in range(50):
        logits, cache = model.run_with_cache(tokens)

        for layer in range(LAYERS):
            attn_dict[layer].append(cache["attn_out", layer][0, -1, :].detach())
            mlp_dict[layer].append(cache["mlp_out", layer][0, -1, :].detach())

        next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=-1)

    attn_dict = {layer: torch.stack(attn_dict[layer]) for layer in range(LAYERS)}
    mlp_dict = {layer: torch.stack(mlp_dict[layer]) for layer in range(LAYERS)}

    prompt_len = model.to_tokens(prompt).shape[1]
    response_tokens = tokens[0, prompt_len:]

    return attn_dict, mlp_dict, model.tokenizer.decode(response_tokens)


if __name__ == "__main__":  # 3. Fix: needs double underscores
    attn_dict, mlp_dict, response = plot_residual_streams(
        "What happens to you if you eat watermelon seeds?"
    )
    attn_serializable = {str(layer): attn_dict[layer].tolist() for layer in attn_dict}
    mlp_serializable = {str(layer): mlp_dict[layer].tolist() for layer in mlp_dict}

    with open(f"{MODEL_NAME.split("/")[1]}_attn.json", "w") as f:
        json.dump(attn_serializable, f)  

    with open(f"{MODEL_NAME.split("/")[1]}_mlp.json", "w") as f:
        json.dump(mlp_serializable, f)
    print("Response:", response)
