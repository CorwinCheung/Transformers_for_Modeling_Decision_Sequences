import os
from collections import Counter

# Define the run number and model number
run_number = '7'      # Replace with your run number
model_number = '10M'    # Replace with your model number

# File paths
ground_truth_path = f'../data/2ABT_logistic_run_{run_number}.txt'
predictions_path = f'Preds_for_{run_number}_with_model_{model_number}.txt'

# Load the ground truth data
with open(ground_truth_path, 'r') as f:
    ground_truth = f.read().replace('\n', '')

# Load the model predictions
with open(predictions_path, 'r') as f:
    predictions = f.read().replace('\n', '')

# Ensure both sequences are of the same length
print(len(ground_truth), len(predictions), "lengths should match!")
min_length = min(len(ground_truth), len(predictions))
ground_truth = ground_truth[:min_length]
predictions = predictions[:min_length]

# Compute accuracy
correct = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
total = len(ground_truth)
accuracy = correct / total

print(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")

# Compute confusion matrix
confusion = Counter()
for gt, pred in zip(ground_truth, predictions):
    confusion[(gt, pred)] += 1

# Display confusion matrix
print("\nConfusion Matrix:")
print("Ground Truth -> Prediction: Count")
sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
for (gt_char, pred_char), count in sorted_confusion:
    print(f"{gt_char} -> {pred_char}: {count}")

# Function to normalize characters ('R' and 'r' as 'R', 'L' and 'l' as 'L')
def normalize_char(c):
    if c in ['R', 'r']:
        return 'R'
    elif c in ['L', 'l']:
        return 'L'
    else:
        return c

# Calculate accuracy ignoring 'R'-'r' and 'L'-'l' differences
def calculate_accuracy_ignore_case(ground_truth, predictions):
    correct = sum(
        normalize_char(gt) == normalize_char(pred) 
        for gt, pred in zip(ground_truth, predictions)
    )
    total = len(ground_truth)
    accuracy = correct / total
    return accuracy

# Compute and print the adjusted accuracy
adjusted_accuracy = calculate_accuracy_ignore_case(ground_truth, predictions)
print(f"\nAdjusted Accuracy (ignoring 'R'-'r' and 'L'-'l' differences): {adjusted_accuracy:.2%}")

# Calculate the percentage of trials where the transformer switches its guess between 'R'/'r' and 'L'/'l'
def calculate_switch_percentage(predictions):
    normalized_preds = [normalize_char(pred) for pred in predictions]
    switches = sum(
        1 for i in range(1, len(normalized_preds)) 
        if normalized_preds[i] != normalized_preds[i - 1]
    )
    total_transitions = len(normalized_preds) - 1
    switch_percentage = (switches / total_transitions) * 100 if total_transitions > 0 else 0
    return switch_percentage

# Compute and print the switch percentage
switch_percentage = calculate_switch_percentage(predictions)
gt_switch_percentage = calculate_switch_percentage(ground_truth)
print(f"\nSwitch Percentage (transformer switches between 'R'/'r' and 'L'/'l'): {switch_percentage:.2f}%")
print(f"\nSwitch Percentage (ground truth switches between 'R'/'r' and 'L'/'l'): {gt_switch_percentage:.2f}%")
