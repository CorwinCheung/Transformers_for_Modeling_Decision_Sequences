import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

from scipy.stats import bootstrap

import utils.file_management as fm
#so that I can import from a directory two levels up
# from evaluation.graph_helper import (calculate_switch_probabilities,
#                                      plot_probabilities,
#                                      plot_switch_probabilities)
from utils.parse_data import align_predictions_with_gt, parse_simulated_data

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../behavior-helpers/')))
from bh.visualization import plot_trials as pts

global rflr


def plot_bpos_behavior_predictions(events, run, *, model_name=None, suffix='v'):
    events = events.rename(columns={
                                'block_position': 'iInBlock',
                                'block_length': 'blockLength',
                                'selected_high': 'selHigh',
                                'switch': 'Switch'
                           })
    bpos = pts.calc_bpos_probs(events, add_cond_cols=['context', 'session'],
                               add_agg_cols=['pred_switch', 'pred_selected_high'])
    fig, axs = pts.plot_bpos_behavior(bpos, hue='context',
                                      plot_features={
                                                'pred_selected_high': ('P(High)', (0, 1)),
                                                'pred_switch': ('P(Switch)', (0, 0.4)),
                                      },
                                      errorbar='se')
    [ax.set(xlim=(-10, 20)) for ax in axs]
    axs[1].set(ylim=(0, 0.2))
    bpos_filename = fm.get_experiment_file(f'pred_{model_name}_bpos_behavior_{suffix}.png', run)
    fig.savefig(bpos_filename, bbox_inches='tight')
    print(f'saved block position behavior to {bpos_filename}')
   

def plot_checkpoint_comparison(run=None, suffix='v'):
    """Plot behavior comparisons across checkpoints and contexts."""
    
    from pathlib import Path
    import glob
    run = run or fm.get_latest_run()
    exp_dir = Path(fm.get_experiment_dir(run))
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(str(exp_dir / "model_*cp*.pth"))
    model_names = [Path(f).stem for f in checkpoint_files]
    
    if not model_names:
        print(f"No checkpoint models found in run {run}")
        return
        
    # Load ground truth data once
    behavior_file = fm.get_experiment_file("behavior_run_{}.txt", run, suffix)
    high_port_file = fm.get_experiment_file("high_port_run_{}.txt", run, suffix)
    context_file = fm.get_experiment_file("context_transitions_run_{}.txt", run, suffix)
    
    gt_events = parse_simulated_data(behavior_file, high_port_file, context_file)
    contexts = sorted(gt_events['context'].unique())
    
    # Create figure with subplots for each context
    fig, axes = plt.subplots(1, len(contexts), figsize=(6*len(contexts), 5))
    if len(contexts) == 1:
        axes = [axes]
    
    # Plot each checkpoint's predictions for each context
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    for model_name, color in zip(model_names, colors):
        # Load predictions for this checkpoint
        pred_file = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}")
        predictions = fm.read_sequence(pred_file)
        events = align_predictions_with_gt(gt_events, predictions)
        
        # Plot for each context
        for ax, context in zip(axes, contexts):
            context_data = events[events['context'] == context]
            bpos = pts.calc_bpos_probs(context_data, 
                                     add_agg_cols=['pred_switch', 'pred_selected_high'])
            
            # Plot P(switch)
            ax.errorbar(bpos['iInBlock'], 
                       bpos['pred_switch']['mean'],
                       yerr=bpos['pred_switch']['sem'],
                       color=color,
                       label=f'CP {model_name.split("_cp")[-1]}')
            
            ax.set_title(f'Context {context}')
            ax.set_xlabel('Position in Block')
            ax.set_ylabel('P(Switch)')
            ax.set_ylim(0, 0.4)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.tight_layout()
    fig_path = fm.get_experiment_file(f'checkpoint_comparison_bpos.png', run)
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Saved checkpoint comparison plot to {fig_path}')


def main(run=None, model_name: str = None, suffix: str = 'v'):
    
    # Files will automatically use latest run if run=None
    run = run or fm.get_latest_run()
    
    # Get model info from metadata
    if model_name is None:
        model_info = fm.parse_model_info(run, model_name=model_name)
        model_name = model_info['model_name']

    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, suffix)
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, suffix)
    context_filename = fm.get_experiment_file("context_transitions_run_{}.txt", run, suffix)
    predictions_filename = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}")

    print(behavior_filename, '\n', high_port_filename, '\n', context_filename)

    assert fm.check_files_exist(behavior_filename, high_port_filename, context_filename, predictions_filename)

    # Parse the ground truth events
    gt_events = parse_simulated_data(behavior_filename, high_port_filename, context_filename)

    predictions = fm.read_sequence(predictions_filename)
    print(f"Number of events: {len(gt_events)}")
    print(f"Number of predictions: {len(predictions)}")

    events = align_predictions_with_gt(gt_events, predictions)

    # Calculate and print the percent of trials with a switch.
    percent_switches = round(events.pred_switch.mean() * 100, 2)
    print(f"Percent of trials with a switch: {percent_switches:.2f}%")

    # Plot block position behavior of the model.
    plot_bpos_behavior_predictions(events, run, model_name=model_name, suffix=suffix)

    # # Calculate switch probabilities
    # sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts = calculate_switch_probabilities(events)

    # # Plot the switch probabilities
    # plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts, f"run_{run}", "../")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    main(run=args.run, model_name=args.model_name)
