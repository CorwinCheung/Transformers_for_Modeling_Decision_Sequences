import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
from utils.parse_data import align_predictions_with_gt, parse_simulated_data


def calculate_accuracy_ignore_case(ground_truth, predictions):
    """Calculate accuracy ignoring 'R'-'r' and 'L'-'l' differences"""
    correct = sum(
        gt.upper() == pred.upper()
        for gt, pred in zip(ground_truth, predictions)
    )
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0


'''Switch calculations should now be done on fully parsed file to handle
session boundaries'''
# def calculate_switch_percentage_with_gt(predictions, ground_truth):
#     """ Calculate switch percentage between 'R'/'r' and 'L'/'l'"""
#     upper_preds = [c.upper() for c in predictions]
#     upper_gt = [c.upper() for c in ground_truth]
#     switches = sum(
#         1 for i in range(1, len(upper_preds))
#         if upper_preds[i] != upper_gt[i - 1]
#     )
#     total_transitions = len(upper_preds) - 1
#     return (switches / total_transitions) * 100 if total_transitions > 0 else 0

# def calculate_switch_percentage_within_gt(ground_truth):
#     upper_gt = [c.upper() for c in ground_truth]
#     switches = sum(
#         1 for i in range(1, len(upper_gt))
#         if upper_gt[i] != upper_gt[i - 1]
#     )
#     total_transitions = len(upper_gt) - 1
#     return (switches / total_transitions) * 100 if total_transitions > 0 else 0


def main(run=None, model_name=None):

    if run is None:
        run = fm.get_latest_run()

    # Get model info from metadata
    model_info = fm.parse_model_info(run, model_name=model_name)
    model_name = model_info['model_name']

    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, 'v')
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, 'v')
    context_filename = fm.get_experiment_file("context_transitions_run_{}.txt", run, 'v')
    predictions_filename = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}")

    print(behavior_filename, '\n', high_port_filename, '\n', context_filename)

    assert fm.check_files_exist(behavior_filename, high_port_filename, context_filename, predictions_filename)

    # Parse the ground truth events and map in predictions
    ground_truth = parse_simulated_data(behavior_filename, high_port_filename, context_filename)    
    predictions = list(fm.read_sequence(predictions_filename))
    aligned_data = align_predictions_with_gt(ground_truth, predictions)

    assert len(ground_truth) == len(predictions), (
        "Ground truth and predictions have different lengths")

    # Compute accuracy of all 4 possible characters
    correct = np.sum(aligned_data['k0'] == aligned_data['pred_k0'])
    total = len(ground_truth)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")

    # Compute confusion matrix
    ground_truth_tokens = list(ground_truth['k0'].values)
    confusion = Counter((gt, pred) for gt, pred in zip(ground_truth_tokens, predictions))

    # Display confusion matrix
    print("\nConfusion Matrix:")
    print("Ground Truth -> Prediction: Count")
    for (gt_char, pred_char), count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        print(f"{gt_char} -> {pred_char}: {count}")

    # Compute and print the adjusted accuracy (choice only scoring)
    choice_accuracy = np.mean(aligned_data['pred_choice'] == aligned_data['choice'])
    print(f"\nChoice only Accuracy ('R'-'r','L'-'l' same): {choice_accuracy:.2%}")

    # Compute and print the accuracy on reward predictions
    reward_accuracy = np.mean(aligned_data['pred_reward'] == aligned_data['reward'])
    print(f"\nReward Accuracy ('R'-'r','L'-'l' same): {reward_accuracy:.2%}")

    # # Compute and print switch percentages
    gt_switch_percentage = round(aligned_data.switch.mean() * 100, 2)
    pred_switch_percentage = round(aligned_data.pred_switch.mean() * 100, 2)
    print(f"\nSwitch Percentage (model): {pred_switch_percentage:.2f}%")
    print(f"Switch Percentage (ground truth): {gt_switch_percentage:.2f}%")

    labels = sorted(set(ground_truth_tokens + predictions))
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
    cm_file = fm.get_experiment_file("cm_pred_run_{}.png", run, f"_{model_name}")
    print(f'saved confusion matrix to {cm_file}')
    plt.savefig(cm_file)

if __name__ == "__main__":
    main()
