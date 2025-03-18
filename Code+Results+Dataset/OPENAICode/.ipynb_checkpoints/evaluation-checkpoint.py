import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data= pd.read_csv("results_data_mistral-7b_fulldataset.csv")
print("Data Length:", len(data))

# Extract true labels and predicted labels

true_labels = data["label"]
predicted_labels = data["mistral_prediction"]

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")