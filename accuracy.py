#!/usr/bin/env python3
import sys
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

if len(sys.argv) != 2:
    print("Usage: {} <results.json>".format(sys.argv[0]))
    sys.exit(1)

filename = sys.argv[1]
with open(filename, "r") as f:
    results = json.load(f)

filtered_true, filtered_pred = [], []
for item in results:
    t = item["true_label"]
    p = item["predicted_label"]
    
    if t == '' or str(p) == '':
        continue

    try:
        t_int = int(t)
        p_int = int(p)
    except ValueError:
        continue

    filtered_true.append(t_int)
    filtered_pred.append(p_int)

accuracy = accuracy_score(filtered_true, filtered_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

precision, recall, f1, _ = precision_recall_fscore_support(
    filtered_true, filtered_pred, average='weighted', zero_division=0
)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(filtered_true, filtered_pred, zero_division=0))

