from ollama import Client
import pandas as pd
import re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
file = f"/Users/rachitdas/Desktop/hallucinations_arc/dev/{re.sub("/"," ",MODEL_NAME)}_without_labels_last.pkl"


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
    - Factually incorrect information
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
    df=pd.read_pickle(file)

    client=Client()
    results=[]

    for i, row in df.iterrows():
        model_output=row['model_output']
        correct_output=row['reference']

        prompt=return_prompt(model_output, correct_output)

        messages=[{
            'role':'user',
            'content':prompt
        }]

        response = client.chat(model="gpt-oss:120b-cloud",messages=messages)

        content=response.message.content
        hallucination_label=content.split("<hallucination>")[1].split("</hallucination>")[0].strip()
        results.append(int(hallucination_label))
        if(i%5==0):
            print(f"Evaluated {i+1}/{len(df)} responses")

    df['hallucination_label']=results
    df.to_pickle(f"{re.sub("/"," ",MODEL_NAME)}_hallucination_states_last.pkl")

if __name__ == "__main__":
    evaluate_responses()
