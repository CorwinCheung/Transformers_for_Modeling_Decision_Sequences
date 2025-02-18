import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

from evaluation.graph_helper import calc_bpos_behavior, plot_bpos_behavior
import utils.file_management as fm

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../behavior-helpers/')))
from bh.visualization import plot_trials as pts

from utils.parse_data import add_sequence_columns, parse_simulated_data

def initialize_logger(run):
    global logger
    logger = fm.setup_logging(run, 'inference', 'learning')

def load_predictions(run=None, model_name=None):

    """Load and process all prediction files for a run efficiently."""
    run = run or fm.get_latest_run()

    # Get model info from metadata.
    model_info = fm.parse_model_info(run, model_name=model_name)
    if model_name is not None:
        assert model_info['model_name'] == model_name, 'did not recover correct model'
    model_name = model_info['model_name']
    pred_file = fm.get_experiment_file(f"learning_{model_name}_val_preds.txt", run, subdir='seqs')

    predictions = pd.read_csv(pred_file, sep='\t')
    predictions = predictions.sort_values(['Step', 'Idx'])

    return predictions, model_info


def load_behavior_data(model_info):
    data_path = model_info['dataloader']['File validated on']
    if not os.path.isfile(data_path):
        data_path = fm.convert_to_local_path(data_path)
    high_port_path = data_path.replace('behavior', 'high_port')
    sessions_path = data_path.replace('behavior', 'session_transitions')
    events = parse_simulated_data(data_path, high_port_path, sessions_path)
    return events


def plot_bpos_behavior_learning(predictions,
                                *,
                                model_name=None,
                                run=None,
                                step_max=None,
                                suffix='v'):
    """Plot behavioral position analysis of model predictions.

    Calculates and plots switch probability and high port selection probability
    as a function of position in block, separated by training steps.

    Args:
        predictions (pd.DataFrame): DataFrame containing model predictions with columns:
            - Step: Training step number
            - True/Predicted: True and predicted choices
            - seqN_RL/seqN: Previous N choices
        model_name (str): Name of the model for saving plot
        run (int): Run number for saving plot

    Returns:
        None. Saves plot to file.
    """
    bpos = calc_bpos_behavior(predictions, add_cond_cols=['Step', 'session'],
                              add_agg_cols=['pred_Switch', 'pred_selHigh'])
    fig, axs = plot_bpos_behavior(bpos, run=run, suffix=suffix,
                                  hue='Step', palette='viridis',
                                  alpha=0.8,
                                  plot_features={
                                    'pred_selHigh': ('P(selHigh)', (0, 1)),
                                    'pred_Switch': ('P(Switch)', (0, 0.3))
                                  },
                                  errorbar=None,
                                  save=False)
    axs[1].get_legend().set(title='Step')
    fig_path = fm.get_experiment_file(f'bpos_{model_name}_{step_max}_{suffix}.png', run, subdir='learning')
    fig.savefig(fig_path)
    logger.info(f'saved bpos plot to {fig_path}')


def plot_conditional_switching_learning(predictions, model_name, seq_length, run, step_max=None):

    # Note, here we can predict on switch because seqN reflects history.
    policies_true = pts.calc_conditional_probs(
        predictions.query('Step==Step.max()'),
        htrials=seq_length, sortby='pevent',
        pred_col='Switch')
    fig, ax = pts.plot_sequences(policies_true)

    policies_pred = pts.calc_conditional_probs(predictions, htrials=seq_length,
                                                sortby='pevent', pred_col='pred_Switch', add_grps='Step')
    fig, ax = pts.plot_sequence_points(policies_pred, grp='Step',
                                        palette='viridis', yval='pevent',
                                        size=3, ax=ax, fig=fig)
    if seq_length > 2:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.get_legend().set(bbox_to_anchor=(1.05, 0), loc='lower left', title='Step')
    fig_path = fm.get_experiment_file(f"learning_{model_name}_val_preds_seq{seq_length}_{step_max}.png", run, subdir='learning')
    fig.savefig(fig_path)
    logger.info(f'saved conditional probabilities for {seq_length} trials to {fig_path}')


def add_choice_metrics(df, prefix=''):
    """Add choice-related metrics with optional prefix for predicted values."""
    source = 'Predicted' if prefix == 'pred_' else 'True'

    # Get previous choice from encoded sequence
    df['prev_choice'] = df['seq2_RL'].apply(lambda x: x[-1].upper()
                                            if not pd.isna(x) else None)

    # Calculate choice metrics
    df[f'{prefix}choice_str'] = df[source].apply(lambda x: x.upper())
    df[f'{prefix}Switch'] = ((df['prev_choice'] != df[f'{prefix}choice_str']).astype(int)
                             .where(df['prev_choice'].notna(), np.nan)
                             .astype('Int64'))

    df[f'{prefix}choice'] = (df[f'{prefix}choice_str'] == 'R').astype(int)
    df[f'{prefix}selHigh'] = (df[f'{prefix}choice'] == df['high_port']).astype(int)

    return df


def preprocess_predictions(predictions, events):

    # Map behavioral data from events DataFrame
    behavioral_cols = {
        'k0': 'k0',  # Verify true choices match
        'block_position': 'iInBlock',
        'block_length': 'blockLength',
        'high_port': 'high_port',
        'domain': 'domain',
        'session': 'session'
    }
    for event_col, pred_col in behavioral_cols.items():
        predictions[pred_col] = predictions['Idx'].map(events[event_col])
    assert all(predictions['k0'] == predictions['True']), "k0 and True values don't match"

    for N in [2, 3]:
        events = add_sequence_columns(events, seq_length=N)
        predictions[f'seq{N}_RL'] = predictions['Idx'].map(events[f'seq{N}_RL'])
        predictions[f'seq{N}'] = predictions['Idx'].map(events[f'seq{N}'])

    # Add metrics for true and predicted choices
    predictions = add_choice_metrics(predictions)  # True choices
    predictions = add_choice_metrics(predictions, prefix='pred_')  # Predicted choices
    return predictions


def main(run=None, model_name=None, step_min=0, step_max=None):

    logger = initialize_logger(run)
    predictions, model_info = load_predictions(run=run, model_name=model_name)
    model_name = model_info['model_name']
    if step_max is None:
        step_max = predictions['Step'].max()

    events = load_behavior_data(model_info)
    predictions = preprocess_predictions(predictions, events)

    predictions = predictions.query('Step.between(@step_min, @step_max)')
    
    if len(predictions) == 0:
        logger.info(f'No steps between {step_min} and {step_max}')
        return None
    run_dir = fm.get_run_dir(run)
    
    os.makedirs(os.path.join(run_dir, 'learning'), exist_ok=True)

    plot_bpos_behavior_learning(predictions, model_name=model_name, run=run, step_max=step_max)

    for N in [2, 3]:
        plot_conditional_switching_learning(predictions, model_name, N, run, step_max=step_max)

if __name__ == "__main__":
    print('-' * 80)
    print('learning.py\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--step_max', type=int, default=None)
    parser.add_argument('--step_min', type=int, default=0)
    args = parser.parse_args()
    main(run=args.run, model_name=args.model_name, step_min=args.step_min, step_max=args.step_max)
