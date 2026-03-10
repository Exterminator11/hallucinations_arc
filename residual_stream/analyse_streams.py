import transformer_lens
import torch
import pandas as pd
import re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformer_lens.HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME, dtype="bfloat16"
)
LAYERS = model.cfg.n_layers

df = pd.read_pickle(
    f"/Users/rachitdas/Desktop/hallucinations_arc/dev/Qwen Qwen2.5-1.5B-Instruct_hallucination_states_last.pkl"
)


def get_streams(prompt, n_tokens=50):
    attn_dict = {l: [] for l in range(LAYERS)}
    mlp_dict = {l: [] for l in range(LAYERS)}
    tokens = model.to_tokens(prompt)

    for _ in range(n_tokens):
        logits, cache = model.run_with_cache(tokens)
        for l in range(LAYERS):
            attn_dict[l].append(cache["attn_out", l][0, -1, :].detach().float().cpu())
            mlp_dict[l].append(cache["mlp_out", l][0, -1, :].detach().float().cpu())
        next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=-1)

    return (
        {l: torch.stack(attn_dict[l]) for l in range(LAYERS)},  # (50, 1536) per layer
        {l: torch.stack(mlp_dict[l]) for l in range(LAYERS)},
    )


# --- Main loop over your labeled dataframe ---
attn_streams = {l: [] for l in range(LAYERS)}
mlp_streams = {l: [] for l in range(LAYERS)}
labels = []

for i, row in df.iterrows():
    prompt = f"Question: {row['question']}\nAnswer:"

    attn_dict, mlp_dict = get_streams(prompt)

    for l in range(LAYERS):
        attn_streams[l].append(attn_dict[l])  # list of (50, 1536)
        mlp_streams[l].append(mlp_dict[l])

    labels.append(row["hallucination_label"])  # already 0 or 1

    if i % 5 == 0:
        print(
            f"[{i+1}/{len(df)}] label={row['hallucination_label']} | {row['model_output'][:60]}"
        )

torch.save(
    {
        "attn_streams": attn_streams,  # dict[layer] → list of N tensors (50, 1536)
        "mlp_streams": mlp_streams,
        "labels": labels,  # list of N ints: 0=truth, 1=hallucination
    },
    f"{re.sub('/', ' ', MODEL_NAME)}_streams_labeled.pt",
)

print("Done!")
