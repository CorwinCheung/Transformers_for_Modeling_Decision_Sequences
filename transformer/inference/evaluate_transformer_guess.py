import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
from utils.parse_data import align_predictions_with_gt, parse_simulated_data

def initialize_logger(run):
    global logger
    logger = fm.setup_logging(run, 'inference', 'evaluate_transformer_guess')

def print_accuracy(aligned_data):

    # Compute accuracy of all 4 possible characters
    correct = np.sum(aligned_data['k0'] == aligned_data['pred_k0'])
    total = len(aligned_data)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\n{' ':>37}Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")

    # Compute and print the adjusted accuracy (choice only scoring)
    choice_accuracy = np.mean(aligned_data['pred_choice'] == aligned_data['choice'])
    logger.info(f"{' ':>10}Choice only Accuracy (Right/Left same): {choice_accuracy:.2%}")

    # Compute and print the accuracy on reward predictions
    reward_accuracy = np.mean(aligned_data['pred_reward'] == aligned_data['reward'])
    logger.info(f"{' ':>15}Reward Accuracy (Upper/Lower same): {reward_accuracy:.2%}")


def plot_confusion_matrix(aligned_data, run, model_name, domain=''):

    ground_truth_tokens = list(aligned_data['k0'].values)
    prediction_tokens = list(aligned_data['pred_k0'].values)
    confusion = Counter((gt, pred) for gt, pred in zip(ground_truth_tokens, prediction_tokens))
    logger.info("\nConfusion Matrix:")
    logger.info("Ground Truth -> Prediction: Count")
    for (gt_char, pred_char), count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{gt_char} -> {pred_char}: {count}")

    labels = sorted(set(ground_truth_tokens + prediction_tokens))
    label_map = {label: i for i, label in enumerate(labels)}
    # Initialize the confusion matrix array
    conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Populate the confusion matrix array
    for (gt_char, pred_char), count in confusion.items():
        i, j = label_map[gt_char], label_map[pred_char]
        conf_matrix[i, j] = count
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted Label", ylabel="Ground Truth Label", title="Confusion Matrix")
    cm_file = fm.get_experiment_file("cm_pred_run_{}.png", run, f"_{model_name}_{domain}", subdir='predictions')
    logger.info(f'saved confusion matrix to {cm_file}')
    fig.savefig(cm_file)


def print_switches(aligned_data):
    # # Compute and print switch percentages
    gt_switch_percentage = round(aligned_data.switch.mean() * 100, 2)
    pred_switch_percentage = round(aligned_data.pred_switch.mean() * 100, 2)
    logger.info(f"\nSwitch Percentage (model): {pred_switch_percentage:.2f}%")
    logger.info(f"Switch Percentage (ground truth): {gt_switch_percentage:.2f}%")


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


def load_data(run, model_name):
    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, 'v', subdir='seqs')
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, 'v', subdir='seqs')
    session_filename = fm.get_experiment_file("session_transitions_run_{}.txt", run, 'v', subdir='seqs')
    predictions_filename = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}", subdir='seqs')

    logger.info(f'behavior_filename: {behavior_filename}')
    logger.info(f'high_port_filename: {high_port_filename}')
    logger.info(f'session_filename: {session_filename}')

    assert fm.check_files_exist(behavior_filename, high_port_filename, session_filename, predictions_filename)

    # Parse the ground truth events and map in predictions
    ground_truth = parse_simulated_data(behavior_filename, high_port_filename, session_filename)    
    predictions = list(fm.read_sequence(predictions_filename))

    assert len(ground_truth) == len(predictions), (
        "Ground truth and predictions have different lengths")

    aligned_data = align_predictions_with_gt(ground_truth, predictions)
    return aligned_data


def main(run=None, model_name=None):

    if run is None:
        run = fm.get_latest_run()

    initialize_logger(run)

    if model_name is None:
        # Get model info from metadata
        model_info = fm.parse_model_info(run, model_name=model_name)
        model_name = model_info['model_name']
    aligned_data = load_data(run, model_name)

    for domain, data in aligned_data.groupby('domain'):
        print(f'\n\nAnalysis for Domain {domain}')
        print_accuracy(data)
        print_switches(data)
        plot_confusion_matrix(data, run, model_name, domain)

if __name__ == "__main__":
    print('-' * 80)
    print('evaluate_transformer_guess.py\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    main(run=args.run, model_name=args.model_name)
