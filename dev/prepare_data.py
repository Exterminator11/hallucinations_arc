from groq import Groq
from dotenv import load_dotenv
import time
from tqdm import tqdm
import os
import random
import re

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def create_unique_facts(number_of_facts:int=25000):
    call_limit=10
    facts_per_call=100
    total_facts_created=0
    for _ in tqdm(range(number_of_facts//(facts_per_call*call_limit))):
        prompt=f"""Generate {facts_per_call} that are unique, and fake. Be as absurd as possible and make sure they are not true. Each fact should be on a new line. Output only the facts without any additional text.
        """
        for _ in range(call_limit):
            response = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates fake facts.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50000,
                temperature=1.2,
            )
            with open("data/unique_facts.txt","a") as f:
                f.write(response.choices[0].message.content+"\n")  # type: ignore
        total_facts_created+=facts_per_call
        time.sleep(60)


def repeat_facts(
    number_of_facts: int = 25000, repeat_percentage: float = 0.10, batch_size: int = 100
):
    """
    Rephrase a percentage of facts in 'data/unique_facts.txt' and append rephrased facts back.
    Processes in batches to handle LLM context limits.
    """
    rephrased_facts_needed = int(number_of_facts * repeat_percentage)

    # Read unique facts
    with open("data/unique_facts.txt", "r", encoding="utf-8") as f:
        unique_facts = [line.strip() for line in f if line.strip()]

    # Process in batches
    for i in tqdm(range(0, rephrased_facts_needed, batch_size)):
        current_batch_size = min(batch_size, rephrased_facts_needed - i)
        batch_facts = random.sample(unique_facts, current_batch_size)

        prompt = f"""
Rephrase the following facts to create new facts. 
Ensure that the rephrased facts are not identical to the originals. 
Each fact should be on a new line. 
Output only the rephrased facts without any additional text.

Facts to rephrase:
{''.join(f'- {fact}\n' for fact in batch_facts)}
"""

        response = groq.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates fake facts.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=32768,
            temperature=1.2,
        )

        # Clean the output
        new_facts_text = response.choices[0].message.content.strip()  # type: ignore
        new_facts_lines = [
            line.strip() for line in new_facts_text.split("\n") if line.strip()
        ]

        # Append to file
        with open("data/unique_facts.txt", "a", encoding="utf-8") as f:
            for fact in new_facts_lines:
                f.write(fact + "\n")


def mix_file():
    with open("data/unique_facts.txt","r") as f:
        all_facts=f.readlines()
    random.shuffle(all_facts)
    with open("data/final_facts_dataset.txt","w") as f:
        f.writelines(all_facts)

def create_dataset():
    create_unique_facts(25000)
    repeat_facts(25000,0.05)
    mix_file()
    clean_file()

def clean_file():

    input_file = "data/final_facts_dataset.txt"
    temp_file = "temp_file.txt"

    with open(input_file, "r", encoding="utf-8") as infile, open(
        temp_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = line.replace("\n", " ").strip()
            if line:
                outfile.write(line + "\n")

    # Replace original file with cleaned file
    os.replace(temp_file, input_file)


if __name__ == "__main__":
    create_dataset()

