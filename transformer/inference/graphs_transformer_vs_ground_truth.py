import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
from evaluation.graph_helper import (calc_bpos_behavior, plot_bpos_behavior,
                                     plot_conditional_switching_eval)
from utils.parse_data import (align_predictions_with_gt, get_data_filenames,
                              load_predictions, parse_simulated_data)


def initialize_logger(run):
    global logger
    logger = fm.setup_logging(run, 'inference', 'graphs_transformer_vs_ground_truth')


def main(run=None, model_name: str = None, suffix: str = 'v'):

    # Files will automatically use latest run if run=None
    run = run or fm.get_latest_run()
    initialize_logger(run)
    # run_dir = fm.get_run_dir(run)
    # os.makedirs(os.path.join(run_dir, 'predictions'), exist_ok=True)

    # Get model info from metadata
    if model_name is None:
        model_info = fm.parse_model_info(run, model_name=model_name)
        model_name = model_info['model_name']
    print("model_name: ", model_name)
    
    events = load_predictions(run, model_name, suffix=suffix)

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
    print("plotting conditional switching eval")
    for seq in [2,3]:
        plot_conditional_switching_eval(events, seq_length=seq, run=run, suffix=f'{model_name}_{suffix}',
                               subdir='predictions')

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
