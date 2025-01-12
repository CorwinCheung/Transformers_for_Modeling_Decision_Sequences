import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Define run number and model number
run_number = '3'
model_name = 'wandb_model_task_782_seen999M'

# File paths
ground_truth_path = f'../../data/2ABT_behavior_run_{run_number}.txt'
predictions_path = f'Preds_for_{run_number}_with_model_{model_name}.txt'

# Load ground truth data
with open(ground_truth_path, 'r') as f:
    ground_truth = f.read().replace('\n', '').replace(' ', '')

# Load model predictions
with open(predictions_path, 'r') as f:
    predictions = f.read().replace('\n', '').replace(' ', '')

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