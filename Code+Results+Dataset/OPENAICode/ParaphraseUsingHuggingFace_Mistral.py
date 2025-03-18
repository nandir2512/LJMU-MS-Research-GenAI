import pandas as pd
from tqdm.auto import tqdm
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load dataset
print("Loading dataset...")
test = pd.read_csv("test.csv")
data = test.copy()

# Initialize Hugging Face Inference Clientl
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",  
    token=HUGGINGFACEHUB_API_TOKEN
)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["sentence1", "sentence2"],
    template="""
    Determine if two sentences convey the same core information. A paraphrase maintains the essential meaning despite different wording.
    Respond with one word: paraphrase or non-paraphrase.

    Now classify:
    Sentence 1: "{sentence1}"
    Sentence 2: "{sentence2}"
    Answer:  """
)

# Function to call the model and check paraphrase
def check_paraphrase(sentence1, sentence2):
    prompt = prompt_template.format_prompt(sentence1=sentence1, sentence2=sentence2).text
    #print(f"\nGenerated Prompt:{prompt}")
    try:
        response = client.text_generation(prompt, max_new_tokens=6, temperature=0.1)
        response_text = response.strip().lower()
        #print(f"Model Response: {response_text}")
        if response_text=="paraphrase":
            binary_label=1
        elif response_text=="non-paraphrase":
            binary_label=0
        else:
            binary_label="None"

        return response_text, binary_label
    except Exception as e:
        print(f"Error processing sentence pair: {e}")
        return None  # Handle failures gracefully
    
# Apply paraphrase checking with progress bar
print("Processing paraphrase detection...")
tqdm.pandas()  # Enable progress bar for Pandas

data[["llama_predict", "llama_prediction"]] = pd.DataFrame(data.progress_apply(
    lambda row: check_paraphrase(row["sentence1"], row["sentence2"]), axis=1).tolist(),
    index=data.index)

# Save results
data.to_csv("results_data.csv", index=False)
print("Results saved to results_data.csv.")

# # Extract true labels and predicted labels
# true_labels = data["label"]
# predicted_labels = data["llama_prediction"]

# # Calculate metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels)
# recall = recall_score(true_labels, predicted_labels)
# f1 = f1_score(true_labels, predicted_labels)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")