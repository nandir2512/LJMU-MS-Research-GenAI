from openai import OpenAI
from datasets import load_dataset
import openai
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
print("Load Dataset")
test =pd.read_csv("test.csv")
data=test.copy()#head(10)
#print(data)

tqdm.pandas()


def check_paraphrase(sentence1, sentence2):
    prompt = f"""
    We have two sentences. Determine if they are paraphrases of each other.
    A paraphrase conveys the same meaning using different words while maintaining the core information. A non-paraphrase has a different meaning or significantly alters the information.  

    Respond strictly with one word: **paraphrase** or **non-paraphrase**. 

    Now classify:
    Sentence 1: "{sentence1}"
    Sentence 2: "{sentence2}"
    Answer:
    """

    response= openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content":prompt},
        ],
        response_format={
        "type": "text"
        },
        max_completion_tokens=4,
        temperature=0.5
    )
    llm_response = response.choices[0].message.content.strip()
    if llm_response=='non-paraphrase':
        return 0
    else:
        return 1

def process_row(row):
    row["OpenAI_prediction"] = check_paraphrase(row["sentence1"], row["sentence2"])
    return row

# Apply function with progress bar
print("Processing Rows...")
data = data.progress_apply(process_row, axis=1)

# data = data.apply(process_row, axis=1)
# print(data)
data.to_csv("GPT-3.5-turbo.csv")



# Extract true labels and predicted labels
true_labels = data["label"]
predicted_labels = data["OpenAI_prediction"]

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")