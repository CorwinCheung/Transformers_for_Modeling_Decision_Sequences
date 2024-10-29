from collections import Counter

# Define run number and model number
run_number = '4'
model_name = '92K'

# File paths
# ground_truth_path = f'../../data/2ABT_logistic_run_{run_number}.txt'
ground_truth_path = '../../data/test.txt'
# predictions_path = f'Preds_for_{run_number}_with_model_{model_name}.txt'
predictions_path = 'Preds_test.txt'

# Load ground truth data
with open(ground_truth_path, 'r') as f:
    ground_truth = f.read().replace('\n', '')
    ground_truth = ground_truth.replace('S', '')
    ground_truth = ground_truth.replace('O', '')

# Load model predictions
with open(predictions_path, 'r') as f:
    predictions = f.read().replace('\n', '')
    predictions = predictions.replace('S', '')
    predictions = predictions.replace('O', '')

# Ensure both sequences have the same length
min_length = min(len(ground_truth), len(predictions))
ground_truth = ground_truth[:min_length]
predictions = predictions[:min_length]

# Compute accuracy
correct = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
total = len(ground_truth)
accuracy = correct / total
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
    return correct / len(ground_truth)

# Compute and print the adjusted accuracy
adjusted_accuracy = calculate_accuracy_ignore_case(ground_truth, predictions)
print(f"\nAdjusted Accuracy (ignoring 'R'-'r' and 'L'-'l' differences): {adjusted_accuracy:.2%}")

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
