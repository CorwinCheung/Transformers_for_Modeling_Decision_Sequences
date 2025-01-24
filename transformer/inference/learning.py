import os
import sys
from pathlib import Path

import pandas as pd

repo_path = Path(__file__).parent.parent.parent
code_path = repo_path.parent
sys.path.append(f'{str(repo_path)}/')
from utils.file_management import (convert_to_local_path, get_experiment_file,
                                   get_latest_run, parse_model_info)

behavior_helpers_path = code_path / 'behavior-helpers'

sys.path.append(f'{str(behavior_helpers_path)}/')
from bh.visualization import plot_trials as pts

from evaluation.graph_helper import map_rl_to_pattern
from evaluation.graphs_on_trial_block_transitions import parse_files


def load_predictions(run=None, model_name=None):

    """Load and process all prediction files for a run efficiently."""
    run = run or get_latest_run()
    # Load predictions    # Get model info from metadata
    model_info = parse_model_info(run, model_name=model_name)
    model_name = model_info['model_name']

    pred_file = get_experiment_file(f"learning_{model_name}_val_preds.txt", run)
    context_file = get_experiment_file(f"learning_{model_name}_val_context.txt", run)

    predictions = pd.read_csv(pred_file, sep='\t')
    # Sort by step and original index
    predictions = predictions.sort_values(['Step', 'Idx'])

    return predictions, model_info


def plot_bpos_behavior_learning(predictions, model_name, run):

    bpos = pts.calc_bpos_probs(predictions, add_cond_cols=['Step'],
                               add_agg_cols=['pred_Switch', 'pred_selHigh'])
    fig, axs = pts.plot_bpos_behavior(bpos, hue='Step', palette='viridis',
                                      alpha=0.3,
                                      plot_features={
                                         'pred_Switch': ('P(Switch)', (0, 0.4)),
                                         'pred_selHigh': ('P(selHigh)', (0, 1))
                                         })
    [ax.set(xlim=(-10, 20)) for ax in axs]
    axs[1].get_legend().set(bbox_to_anchor=(1.05, 0), loc='lower left', title='Step')
    fig_path = get_experiment_file(f"learning_{model_name}_val_preds_bpos.png", run)
    fig.savefig(fig_path)
    print(f'saved bpos plot to {fig_path}')


def add_choice_metrics(df, prefix=''):
    """Add choice-related metrics with optional prefix for predicted values."""
    source = 'Predicted' if prefix == 'pred_' else 'True'

    # Get previous choice from context
    df['prev_choice'] = df['seq3_RL'].apply(lambda x: x[-1].upper())

    # Calculate choice metrics
    df[f'{prefix}choice_str'] = df[source].apply(lambda x: x.upper())
    df[f'{prefix}Switch'] = (df['prev_choice'] != df[f'{prefix}choice_str']).astype(int)
    df[f'{prefix}choice'] = (df[f'{prefix}choice_str'] == 'R').astype(int)
    df[f'{prefix}selHigh'] = (df[f'{prefix}choice'] == df['high_port']).astype(int)

    return df


def main(run=None, model_name=None):

    predictions, model_info = load_predictions()
    try:
        data_path = model_info['dataloader']['File validated on']
        high_port_path = data_path.replace('behavior', 'high_port')
        events = parse_files(data_path, high_port_path)
    except FileNotFoundError:
        data_path = convert_to_local_path(data_path)
        high_port_path = data_path.replace('behavior', 'high_port')
        events = parse_files(data_path, high_port_path)

    # Map behavioral data from events DataFrame
    behavioral_cols = {
        'k0': 'k0',  # Verify true choices match
        'block_position': 'iInBlock',
        'block_length': 'blockLength',
        'high_port': 'high_port'
    }
    for event_col, pred_col in behavioral_cols.items():
        predictions[pred_col] = predictions['Idx'].map(events[event_col])
    assert all(predictions['k0'] == predictions['True']), "k0 and True values don't match"

    for N in [2, 3]:
        events[[f'seq{N}_RL', f'seq{N}']] = None
        events.loc[N:, f'seq{N}_RL'] = [''.join(events['k0'].values[i-N:i])
                                        for i in range(N, len(events))]
        events.loc[N:, f'seq{N}'] = events.loc[N:, f'seq{N}_RL'].apply(map_rl_to_pattern)
        predictions[f'seq{N}_RL'] = predictions['Idx'].map(events[f'seq{N}_RL'])
        predictions[f'seq{N}'] = predictions['Idx'].map(events[f'seq{N}'])

    # Add metrics for true and predicted choices
    predictions = add_choice_metrics(predictions)  # True choices
    predictions = add_choice_metrics(predictions, prefix='pred_')  # Predicted choices

    plot_bpos_behavior_learning(predictions, model_name=model_info['model_name'], run=run)

    for N in [2, 3]:
        policies_true = pts.calc_conditional_probs(
            predictions.query('Step==Step.max()'),
            htrials=N, sortby='pevent',
            pred_col='Switch')
        fig, ax = pts.plot_sequences(policies_true)

        policies_pred = pts.calc_conditional_probs(predictions, htrials=N,
                                                   sortby='pevent', pred_col='pred_Switch', add_grps='Step')
        fig, ax = pts.plot_sequence_points(policies_pred, grp='Step',
                                           palette='viridis', yval='pevent',
                                           size=3, ax=ax, fig=fig)
        if N > 2:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig_path = get_experiment_file(f"learning_{model_info['model_name']}_val_preds_seq{N}.png", run)
        fig.savefig(fig_path)
        print(f'saved conditional probabilities for {N} trials to {fig_path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    main(run=args.run, model_name=args.model_name)
