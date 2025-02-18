import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
from evaluation.graph_helper import calc_bpos_behavior, plot_bpos_behavior
from utils.parse_data import align_predictions_with_gt, parse_simulated_data

def initialize_logger(run):
    global logger
    logger = fm.setup_logging(run, 'inference', 'graphs_transformer_vs_ground_truth')

def main(run=None, model_name: str = None, suffix: str = 'v'):
    
    # Files will automatically use latest run if run=None
    run = run or fm.get_latest_run()
    initialize_logger(run)
    run_dir = fm.get_run_dir(run)
    os.makedirs(os.path.join(run_dir, 'predictions'), exist_ok=True)

    # Get model info from metadata
    if model_name is None:
        model_info = fm.parse_model_info(run, model_name=model_name)
        model_name = model_info['model_name']

    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, suffix, subdir='seqs')
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, suffix, subdir='seqs')
    session_filename = fm.get_experiment_file("session_transitions_run_{}.txt", run, suffix, subdir='seqs')
    predictions_filename = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}", subdir='seqs')

    logger.info(f'behavior_filename: {behavior_filename}')
    logger.info(f'high_port_filename: {high_port_filename}')
    logger.info(f'session_filename: {session_filename}')
    logger.info(f'predictions_filename: {predictions_filename}')

    assert fm.check_files_exist(behavior_filename, high_port_filename, session_filename, predictions_filename)

    # Parse the ground truth events
    gt_events = parse_simulated_data(behavior_filename, high_port_filename, session_filename)

    predictions = fm.read_sequence(predictions_filename)
    logger.info(f"Number of events: {len(gt_events)}")
    logger.info(f"Number of predictions: {len(predictions)}")

    events = align_predictions_with_gt(gt_events, predictions)

    # Calculate and print the percent of trials with a switch.
    percent_switches = round(events.pred_switch.mean() * 100, 2)
    logger.info(f"Percent of trials with a switch: {percent_switches:.2f}%")

    # Plot block position behavior of the model.
    bpos = calc_bpos_behavior(events, add_cond_cols=['domain', 'session'],
                              add_agg_cols=['pred_switch', 'pred_selected_high'])
    plot_bpos_behavior(bpos, run, suffix=f'{model_name}_{suffix}',
                       subdir='predictions',
                       hue='domain',
                       plot_features={
                            'pred_selected_high': ('P(High)', (0, 1)),
                            'pred_switch': ('P(Switch)', (0, 0.4)),
                        })

    # # Calculate switch probabilities
    # sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts = calculate_switch_probabilities(events)

    # # Plot the switch probabilities
    # plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts, f"run_{run}", "../")

if __name__ == "__main__":
    print('-' * 80)
    print('graphs_transformer_vs_ground_truth.py\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    main(run=args.run, model_name=args.model_name)
