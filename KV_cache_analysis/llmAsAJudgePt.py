from ollama import Client
import torch
import re

MODEL_NAME = "meta-llama_Llama-3.1-8B-Instruct"
file = "/Users/rachitdas/Desktop/hallucinations_arc/KV_cache_analysis/meta-llama_Llama-3.1-8B-Instruct_all_layers_with_attn.pt"  # updated file


def return_prompt(MODEL_OUTPUT, CORRECT_OUTPUT):
    prompt = f"""
    <prompt>
    You are an expert evaluator tasked with determining whether a language model's output contains hallucinations. A hallucination occurs when the model generates information that is factually incorrect, fabricated, or not supported by the correct output.
    <input>
    <model_output>
{MODEL_OUTPUT}
    </model_output>
    <correct_output>
{CORRECT_OUTPUT}
    </correct_output>
    </input>
    Compare the model output against the correct output. Determine if the model output contains any:
    - Factually incorrect information when compared to the correct output
    - Fabricated details not present in the correct output
    - Contradictions to the correct output
    Note: Paraphrasing, different wording, or omissions are NOT hallucinations.
    Respond with only:
    - 0 if there is NO hallucination
    - 1 if there IS a hallucination
    <hallucination>{{0 or 1}}</hallucination>
    </prompt>
    """
    return prompt


def evaluate_responses():
    records = torch.load(
        file, weights_only=False
    )  # weights_only=False for complex dicts
    client = Client()
    results = []

    for i, record in enumerate(records):
        model_output = record["metadata"]["model_answer"]
        correct_output = record["metadata"]["answer"]

        prompt = return_prompt(model_output, correct_output)
        messages = [{"role": "user", "content": prompt}]

        # Call judge
        response = client.chat(model="gpt-oss:120b-cloud", messages=messages)
        content = response.message.content

        # Parse label
        try:
            hallucination_label = int(
                content.split("<hallucination>")[1].split("</hallucination>")[0].strip() #type: ignore
            )
        except (IndexError, ValueError):
            print(f"  [WARNING] Could not parse label for record {i}, defaulting to -1")
            hallucination_label = -1

        # Add label to metadata
        record["metadata"]["hallucination_label"] = hallucination_label
        results.append(hallucination_label)

        if i % 5 == 0:
            print(f"Evaluated {i+1}/{len(records)} responses")
            print(f"  Q      : {record['metadata']['question'][:60]}")
            print(f"  Answer : {correct_output[:60]}")
            print(f"  Model  : {model_output[:60]}")
            print(f"  Label  : {hallucination_label}")

    # Save back with labels
    save_path = f"{re.sub(r'[/]', '_', MODEL_NAME)}_hallucination_labels_KV.pt"
    torch.save(records, save_path)

    # Summary (exclude failed parses)
    valid = [r for r in results if r != -1]
    n_hallucinations = sum(valid)
    total = len(valid)

    print(f"\nSaved {len(records)} records to {save_path}")
    print(f"Hallucinations : {n_hallucinations}/{total} = {n_hallucinations/total:.1%}")
    print(f"Parse failures : {results.count(-1)}")


if __name__ == "__main__":
    evaluate_responses()
