import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.file_management import (get_experiment_file, get_latest_run,
                                   parse_model_info, read_sequence)


def calculate_accuracy_ignore_case(ground_truth, predictions):
    """Calculate accuracy ignoring 'R'-'r' and 'L'-'l' differences"""
    correct = sum(
        gt.upper() == pred.upper()
        for gt, pred in zip(ground_truth, predictions)
    )
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0


def calculate_switch_percentage_with_gt(predictions, ground_truth):
    """ Calculate switch percentage between 'R'/'r' and 'L'/'l'"""
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


def main(run=None, model_name=None):

    if run is None:
        run = get_latest_run()

    # Get model info from metadata
    model_info = parse_model_info(run, model_name=model_name)
    model_name = model_info['model_name']

    # Load ground truth data
    ground_truth_file = get_experiment_file("behavior_run_{}.txt", run, 'v')
    ground_truth = read_sequence(ground_truth_file)

    # Load model predictions
    pred_file = get_experiment_file("pred_run_{}.txt", run, f"_{model_name}")
    predictions = read_sequence(pred_file)

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

    # Compute and print the adjusted accuracy
    adjusted_accuracy = calculate_accuracy_ignore_case(ground_truth, predictions)
    print(f"\nAdjusted Accuracy ('R'-'r','L'-'l' same): {adjusted_accuracy:.2%}")

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
    cm_file = get_experiment_file("cm_pred_run_{}.png", run, f"_{model_name}")
    plt.savefig(cm_file)

if __name__ == "__main__":
    main()
