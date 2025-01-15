import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Define run number and model number
run_number = 1
model_name = f"sweep_seen9M_run{run_number}"

# File paths
root = os.path.dirname(os.path.dirname(__file__))
ground_truth_path = os.path.join(os.path.dirname(root), 'data', f'2ABT_behavior_run_{run_number}v.txt')
predictions_path = os.path.join(os.path.dirname(__file__), 'predicted_seqs', f'Preds_model_{model_name}.txt')

def read_file(path):
    with open(path, 'r') as f:
        seq = f.read().replace('\n', '').replace(' ', '')
    return seq

# Load ground truth data
ground_truth = read_file(ground_truth_path)

# Load model predictions
predictions = read_file(predictions_path)

# Ensure both sequences have the same length
min_length = min(len(ground_truth), len(predictions))
ground_truth = ground_truth[:min_length]
predictions = predictions[:min_length]

# Compute accuracy of all 4 possible characters
correct = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
total = len(ground_truth)
accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")

# Compute confusion matrix
confusion = Counter((gt, pred) for gt, pred in zip(ground_truth, predictions))

# Display confusion matrix
print("\nConfusion Matrix:")
print("Ground Truth -> Prediction: Count")
for (gt_char, pred_char), count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
    print(f"{gt_char} -> {pred_char}: {count}")

# Calculate accuracy ignoring 'R'-'r' and 'L'-'l' differences
def calculate_accuracy_ignore_case(ground_truth, predictions):
    correct = sum(
        gt.upper() == pred.upper()
        for gt, pred in zip(ground_truth, predictions)
    )
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0

# Compute and print the adjusted accuracy
adjusted_accuracy = calculate_accuracy_ignore_case(ground_truth, predictions)
print(f"\nAdjusted Accuracy ('R'-'r','L'-'l' same): {adjusted_accuracy:.2%}")

# Calculate switch percentage between 'R'/'r' and 'L'/'l'
def calculate_switch_percentage_with_gt(predictions, ground_truth):
    upper_preds = [c.upper() for c in predictions]
    upper_gt = [c.upper() for c in ground_truth]
    switches = sum(
        1 for i in range(1, len(upper_preds))
        if upper_preds[i] != upper_gt[i - 1]
    )
    total_transitions = len(upper_preds) - 1
    return (switches / total_transitions) * 100 if total_transitions > 0 else 0

def calculate_switch_percentage_within_gt(ground_truth):
    upper_gt = [c.upper() for c in ground_truth]
    switches = sum(
        1 for i in range(1, len(upper_gt))
        if upper_gt[i] != upper_gt[i - 1]
    )
    total_transitions = len(upper_gt) - 1
    return (switches / total_transitions) * 100 if total_transitions > 0 else 0

# Compute and print switch percentages
switch_percentage = calculate_switch_percentage_with_gt(predictions, ground_truth)
gt_switch_percentage = calculate_switch_percentage_within_gt(ground_truth)
print(f"\nSwitch Percentage (model): {switch_percentage:.2f}%")
print(f"Switch Percentage (ground truth): {gt_switch_percentage:.2f}%")

labels = sorted(set(ground_truth + predictions))
label_map = {label: i for i, label in enumerate(labels)}

# Initialize the confusion matrix array
conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

# Populate the confusion matrix array
for (gt_char, pred_char), count in confusion.items():
    i, j = label_map[gt_char], label_map[pred_char]
    conf_matrix[i, j] = count

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("Ground Truth Label")
plt.title("Confusion Matrix")
plt.savefig("Conf_matrix_low_val")