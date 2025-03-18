import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data= pd.read_csv("c:\\Users\\USER\\Downloads\\mrpc_test_results-mistral_02.csv")
model_name="Mistral7b"
temp=0.2

print("Data Length:", len(data))

# Extract true labels and predicted labels

true_labels = data["label"]
predicted_labels = data["LLM_response"]

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Model Name:{model_name}, Temp={temp}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")