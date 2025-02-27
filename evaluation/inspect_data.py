import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
import utils.file_management as fm
from utils.parse_data import parse_simulated_data, get_data_filenames, add_sequence_columns, align_predictions_with_gt

logger = None

def initialize_logger(run_number):
    global logger
    logger = fm.setup_logging(run_number, 'data_generation', 'inspect_data')

def compare_train_val(run, train_events, val_events, fig=None, axs=None):

    T_overlap_normalized = {}
    T_overlap_raw = {}
    for T in range (2, 13):
        train_events = add_sequence_columns(train_events, T)
        val_events = add_sequence_columns(val_events, T)

        unique_train_sequences = set(train_events[f'seq{T}_RL'])
        unique_val_sequences = set(val_events[f'seq{T}_RL'])

        overlap = unique_train_sequences.intersection(unique_val_sequences)
        overlap_percentage = len(overlap) / len(unique_val_sequences) * 100
        T_overlap_normalized[T] = overlap_percentage

        raw_overlap = val_events.query(f'seq{T}_RL in @overlap')
        T_overlap_raw[T] = len(raw_overlap) / len(val_events) * 100

        val_events[f'seq{T}_overlap'] = val_events[f'seq{T}_RL'].isin(overlap)

    # Convert dictionaries to DataFrames for proper plotting
    overlap_norm_df = pd.DataFrame({'Sequence Length': list(T_overlap_normalized.keys()), 
                                   'Overlap (%)': list(T_overlap_normalized.values())})
    
    overlap_raw_df = pd.DataFrame({'Sequence Length': list(T_overlap_raw.keys()), 
                                  'Overlap (%)': list(T_overlap_raw.values())})

    if fig is None:
        fig, axs = plt.subplots(ncols=2, figsize=(8,3), layout='constrained')
    
    sns.barplot(x='Sequence Length', y='Overlap (%)', data=overlap_norm_df, ax=axs[0])
    axs[0].set(xlabel='sequence length', ylabel=r'($\%$) overlap', title='Overlap of training and validation sequences\nAs percentage of validation SEQUENCES');

    sns.barplot(x='Sequence Length', y='Overlap (%)', data=overlap_raw_df, ax=axs[1])   
    axs[1].set(xlabel='sequence length', ylabel=r'($\%$) overlap', title='Overlap of training and validation sequences\nAs percentage of validation DATASET');
    sns.despine()
    fig.savefig(fm.get_experiment_file("dataset_overlap.png", run, subdir='agent_behavior'))

    logger.info(f"Total T=12 sequences in training data: {train_events[f'seq12_RL'].nunique()}")
    logger.info(f"Total T=12 sequences in validation data: {val_events[f'seq12_RL'].nunique()}")

    return fig, axs, val_events

def compare_model_performance(run, train_events, val_events, model_name=None):

    if model_name is None:
        # Get model info from metadata
        model_info = fm.parse_model_info(run, model_name=model_name)
        model_name = model_info['model_name']

    predictions_filename = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}", subdir='seqs')
    assert fm.check_files_exist(predictions_filename)

    # Parse the ground truth events and map in predictions
    predictions = list(fm.read_sequence(predictions_filename))

    assert len(val_events) == len(predictions), (
        "Ground truth and predictions have different lengths")
    aligned_data = align_predictions_with_gt(val_events, predictions)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 3), layout='constrained')   
    fig, axs, aligned_data = compare_train_val(run, train_events, aligned_data, fig=fig, axs=axs)
    logger.info(f"Overall prediction accuracy: {aligned_data['pred_correct_k0'].mean() * 100:.2f}%")

    T_accuracy_unique = {}
    for T in range (2, 13):
        T_accuracy_unique[T] = aligned_data.query(f'seq{T}_overlap == True')['pred_correct_k0'].mean()

    # Convert to DataFrame for plotting
    accuracy_df = pd.DataFrame({
        'Sequence Length': list(T_accuracy_unique.keys()), 
        'Accuracy (%)': list(T_accuracy_unique.values())})
    # Keep the existing plots in axs[0] and axs[1]
    sns.barplot(x='Sequence Length', y='Accuracy (%)', data=accuracy_df, ax=axs[-1])
    axs[-1].axhline(y=aligned_data['pred_correct_k0'].mean() * 100, color='k', linestyle='--')
    axs[-1].set(xlabel='sequence length', ylabel='prediction accuracy (%)', 
            title='Model prediction accuracy\nby sequence uniqueness')
    sns.despine()
    fig.savefig(fm.get_experiment_file(f"model_performance_{model_name}.png", run, subdir='models'))

def inspect_batches():
    pass

def main(run=None, model_name=None):

    initialize_logger(run)

    train_events = parse_simulated_data(*get_data_filenames(run, suffix='tr'))
    val_events = parse_simulated_data(*get_data_filenames(run, suffix='v'))
    try:
        compare_model_performance(run, train_events, val_events, model_name)
    except AssertionError:
        _ = compare_train_val(run, train_events, val_events)

if __name__ == "__main__":

    print('-' * 80)
    print('inspect_data.py\n')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)