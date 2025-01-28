import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

from utils.file_management import get_experiment_file

code_path = Path(__file__).parent.parent.parent
behavior_helpers_path = code_path / 'behavior-helpers'
sys.path.append(f'{str(behavior_helpers_path)}/')
import seaborn as sns
from bh.visualization import plot_trials as pts

sns.set_theme(style='ticks', font_scale=1.0, rc={'axes.labelsize': 12,
              'axes.titlesize': 12, 'savefig.transparent': False})

def plot_probabilities(events, run, suffix: str = 'v'):

    events = events.rename(columns={'block_position': 'iInBlock',
                                    'block_length': 'blockLength',
                                    'selected_high': 'selHigh',
                                    'switch': 'Switch'})
    bpos = pts.calc_bpos_probs(events, add_cond_cols=['context', 'session'])
    fig, axs = pts.plot_bpos_behavior(bpos, hue='context', errorbar='se')
    [ax.set(xlim=(-10, 20)) for ax in axs]
    axs[1].set(ylim=(0, 0.2))
    bpos_filename = get_experiment_file('bpos_behavior_{}.png', run, suffix)
    fig.savefig(bpos_filename, bbox_inches='tight')
    print(f'saved block position behavior to {bpos_filename}')


def map_sequence_to_pattern(seq):
    """Maps a sequence of actions to a pattern string (encoding).

    Takes a sequence of actions (dictionaries containing choice and reward
    info) and converts it into a pattern string using the following rules:
    - First action: 'A' if rewarded, 'a' if unrewarded
    - Subsequent actions relative to first choice:
        - Same side as first: 'A' if rewarded, 'a' if unrewarded
        - Different side: 'B' if rewarded, 'b' if unrewarded

    Args:
        seq: Dictionary or dataframe, containing:
            - choice_str: String indicating choice ('L' or 'R')
            - rewarded: Boolean indicating if choice was rewarded

    Returns:
        str: Pattern string encoding the sequence (e.g. 'aAb')
    """
    action1, *actionN = seq

    # First action: 'A' if rewarded, 'a' if unrewarded
    first_letter = 'A' if action1['rewarded'] else 'a'
    first_choice = action1['choice_str']
    pattern = first_letter
    # Subsequent actions

    for i_action in actionN:
        same_side = i_action['choice_str'] == first_choice
        if same_side:
            next_letter = 'A' if i_action['rewarded'] else 'a'
        else:
            next_letter = 'B' if i_action['rewarded'] else 'b'
        pattern += next_letter

    return pattern


def map_rl_to_pattern(seq):
    """Maps a sequence of actions to a pattern string (encoding).

    Takes a sequence of actions already encoded as ['R', 'r', 'L', 'L'] and
    converts it into a pattern string using the following rules:
    - First action: 'A' if rewarded, 'a' if unrewarded
    - Subsequent actions relative to first choice:
        - Same side as first: 'A' if rewarded, 'a' if unrewarded
        - Different side: 'B' if rewarded, 'b' if unrewarded

    Args:
        seq: List, tuple, or string, containing at least two characters.
        Expects encoded actions/outcomes as ['R', 'r', 'L', 'L'].

    Returns:
        str: Pattern string encoding the sequence (e.g. 'aAb')
    """
    action1, *actionN = seq
    first_letter = 'A' if action1.isupper() else 'a'
    first_choice = action1.upper()
    pattern = first_letter
    for i_action in actionN:
        same_side = i_action.upper() == first_choice
        if same_side:
            next_letter = 'A' if i_action.isupper() else 'a'
        else:
            next_letter = 'B' if i_action.isupper() else 'b'
        pattern += next_letter
    return pattern


def add_sequence_columns(events, seq_length):
    """Add sequence columns (history up to current trial) to events DataFrame.

    For a given sequence length N, adds two columns to track trial histories:
    - seqN_RL: right/left encoded sequence of choices/rewards for previous N
               trials (e.g. 'RrL')
    - seqN: Pattern-encoded sequence using A/a/B/b notation (e.g. 'aAb')

    Args:
        events (pd.DataFrame): DataFrame containing trial data with column 'k0' 
            encoding choices/rewards
        seq_length (int): Number of previous trials to include in sequence

    Returns:
        pd.DataFrame: Original DataFrame with added sequence columns. seqN
        is pattern of N previous trials UP TO but NOT INCLUDING the current
        trial.
    """

    events[[f'seq{seq_length}_RL', f'seq{seq_length}']] = None

    # Group by session and calculate sequences
    def get_session_sequences(session_events):
        if len(session_events) >= seq_length:
            start_idx = session_events.index[0] + seq_length
            session_events.loc[start_idx:, f'seq{seq_length}_RL'] = [
                ''.join(session_events['k0'].values[i-seq_length:i])
                for i in range(seq_length, len(session_events))
            ]
            session_events.loc[start_idx:, f'seq{seq_length}'] = (
                session_events.loc[start_idx:, f'seq{seq_length}_RL']
                .apply(map_rl_to_pattern)
            )
        return session_events

    events = events.groupby('session', group_keys=False).apply(get_session_sequences)

    return events


def plot_conditional_switching(events, seq_length, run, suffix: str = 'v'):


    for context in events.context.unique():
        policies = pts.calc_conditional_probs(
            events.query('context == @context'), htrials=seq_length, sortby='pevent', pred_col='switch')
        fig, ax = pts.plot_sequences(policies)

        if seq_length > 2:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        fig_path = get_experiment_file(f"cond_switch_{seq_length}{suffix}{context}.png", run)
        fig.savefig(fig_path)
        print(f'saved conditional probabilities for {seq_length} trials to {fig_path}')

    if events.context.nunique() > 1:
        fig, ax, _ = pts.plot_seq_bar_and_points(events, seq_length,
                                        grp='context',
                                        palette='deep',
                                        pred_col='switch')
        fig_path = get_experiment_file(f"cond_switch_{seq_length}{suffix}_context_comparison.png", run)
        fig.savefig(fig_path)
        print(f'saved context comparison for {seq_length} trials to {fig_path}')
