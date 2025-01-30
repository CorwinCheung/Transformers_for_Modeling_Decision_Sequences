import itertools
import os
# Add the project root directory to Python path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.graph_helper import (add_sequence_columns, plot_bpos_behavior,
                                     plot_conditional_switching)
from utils.file_management import (check_files_exist, get_experiment_file,
                                   read_sequence)

global rflr


def parse_files(behavior_filename, high_port_filename, context_filename, clip_short_blocks=False):

    behavior_data = read_sequence(behavior_filename)
    high_port_data = read_sequence(high_port_filename)
    assert len(behavior_data) == len(high_port_data), (
        "Error: Behavior data and high port data have different lengths.")

    token = list(behavior_data)
    choices = [int(c in ['R', 'r']) for c in behavior_data]
    choices_str = [c.upper() for c in behavior_data]
    rewards = [int(c.isupper()) for c in behavior_data]
    trials = np.arange(len(choices)).astype('int')
    high_ports = [int(hp) for hp in high_port_data]
    selected_high = [c == hp for c, hp in zip(choices, high_ports)]
    switch = np.abs(np.diff(choices, prepend=np.nan))
    transitions = np.abs(np.diff(high_ports, prepend=np.nan))

    events = pd.DataFrame(data={
        'trial_number': trials,
        'k0': token,
        'choice': choices,
        'choice_str': choices_str,
        'reward': rewards,
        'selected_high': selected_high,
        'switch': switch,
        'transition': transitions,
        'high_port': high_ports,
    })

    context_df = pd.read_csv(context_filename, names=['trial_number', 'context'])
    # Forward fill contexts for all trials
    full_context = pd.merge_asof(
        events[['trial_number']],
        context_df.assign(session=np.arange(len(context_df))),
        on='trial_number',
        direction='backward'
    )
    events['context'] = full_context['context']
    events['session'] = full_context['session']

    # First trial in a session cannot be a switch or transition.
    events.loc[events['trial_number'].isin(context_df['trial_number']), 'switch'] = np.nan
    events.loc[events['trial_number'].isin(context_df['trial_number']), 'transition'] = np.nan

    events = get_block_positions(events)
    events = add_sequence_columns(events, seq_length=2)
    events = add_sequence_columns(events, seq_length=3)

    if clip_short_blocks:
        # ID of short blocks, and blocks immediately following short blocks (not back to baseline).
        short_blocks = events.query('block_length < 20')['block_id'].unique()
        post_short_blocks = short_blocks + 1
        raise NotImplementedError
        # For reference:
        # sns.lineplot(data=events.query('block_position.between(0, 20) & ~block_id.isin(@post_short_blocks)'), x='block_position', y='switch', ax=ax)
        # sns.lineplot(data=events.query('rev_block_position.between(-10, 1) & block_length > 20'), x='rev_block_position', y='switch', ax=ax)

    return events


def get_block_positions(events):

    """Calculate block-related metrics, treating each session independently."""
    # Group by session and calculate block information
    session_col = events['session'].copy()
    events_with_blocks = (events
                          .groupby('session', group_keys=False)
                          .apply(lambda session_events: _get_session_block_positions(session_events), include_groups=False))
    events_with_blocks['session'] = session_col


    return events_with_blocks


def _get_session_block_positions(session_events):
    """Helper function to calculate block positions within a single session."""
    # Get transition points within this session

    first_trial = session_events.iloc[0]['trial_number']
    transition_points = session_events.query('transition == 1')['trial_number'].values - first_trial

    # Calculate block lengths
    if len(transition_points) == 0:
        # Single block session
        block_lengths = [len(session_events)]
    else:
        # First block
        block_lengths = [transition_points[0]]
        # Middle blocks
        block_lengths.extend(np.diff(transition_points).astype('int'))
        # Last block
        block_lengths.extend([len(session_events) - transition_points[-1]])
    
    # Calculate length of each block as num trial from previous transition.
    # Prepend first block, and last block is distance to end of sequence.
    # block_lengths = [events.query('transition == 1')['trial_number'].values[0]]
    # block_lengths.extend(events.query('transition == 1')['trial_number'].diff().values[1:].astype('int'))
    # block_lengths.extend([len(events) - events.query('transition == 1')['trial_number'].values[-1]])

    # Store block lengths at transitions and fill backwards (so each trial can
    # reference ultimate block length).
    session_events.loc[session_events.index[0], 'block_length'] = block_lengths[0]
    if len(transition_points) > 0:
        session_events.loc[session_events['transition'] == 1, 'block_length'] = block_lengths[1:]
    session_events['block_length'] = session_events['block_length'].ffill()

    # Counter for index position within each block. Forward and reverse
    # (negative, from end of block backwards).
    block_positions = list(itertools.chain(*[np.arange(i) for i in block_lengths]))
    session_events['block_position'] = block_positions
    session_events['rev_block_position'] = session_events['block_position'] - session_events['block_length']

    # Unique ID for each block.
    if len(transition_points) > 0:
        session_events.loc[session_events.index[0], 'block_id'] = 0
        session_events.loc[session_events['transition'] == 1, 'block_id'] = np.arange(1, len(block_lengths))
        session_events['block_id'] = session_events['block_id'].ffill()

    else:
        session_events['block_id'] = 0
    return session_events


def main(run=None, suffix: str = 'v'):

    # Get file paths using the new utility
    behavior_filename = get_experiment_file("behavior_run_{}.txt", run, suffix)
    high_port_filename = get_experiment_file("high_port_run_{}.txt", run, suffix)
    context_filename = get_experiment_file("context_transitions_run_{}.txt", run, suffix)
    print(behavior_filename, '\n', high_port_filename, '\n', context_filename)

    assert check_files_exist(behavior_filename, high_port_filename, context_filename)

    # Parse the files
    events = parse_files(behavior_filename, high_port_filename, context_filename)
    if events is not None:
        # Calculate and print the percent of trials with a switch
        percent_switches = events['switch'].mean()*100
        print(f"Percent of trials with a switch: {percent_switches:.2f}%")

        # Calculate probabilities for block positions
        plot_bpos_behavior(events, run, suffix=suffix)
        plot_conditional_switching(events, seq_length=2, run=run, suffix=suffix)
        plot_conditional_switching(events, seq_length=3, run=run, suffix=suffix)


# Main code
if __name__ == "__main__":
    main()
