
import itertools

import numpy as np
import pandas as pd

import utils.file_management as fm


def parse_simulated_data(behavior_filename, high_port_filename, context_filename, clip_short_blocks=False):
    """Parse simulated data from behavior, high port, and context files.

    Args:
        behavior_filename (str): Path to behavior file.
        high_port_filename (str): Path to high port file.
        context_filename (str): Path to context file.
        clip_short_blocks (bool): Whether to clip short blocks.

    Returns:
        pd.DataFrame: Parsed data with added sequence columns.

    Notes: Typically for simulated data used as input for training model.
    """

    behavior_data = fm.read_sequence(behavior_filename)
    high_port_data = fm.read_sequence(high_port_filename)
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
        # short_blocks = events.query('block_length < 20')['block_id'].unique()
        # post_short_blocks = short_blocks + 1
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


def align_predictions_with_gt(events, predictions):
    """Align predictions from trained model (e.g. transformer) with ground
    truth events.

    Args:
        events (pd.DataFrame): Ground truth events.
        predictions (list or str): Predictions.

    Returns:
        pd.DataFrame: Aligned events with predictions.
    """

    events_ = events.copy()
    events_['pred_k0'] = [p for p in predictions]
    events_['pred_choice'] = [int(c in ['R', 'r']) for c in predictions]
    events_['pred_choice_str'] = [c.upper() for c in predictions]
    events_['pred_reward'] = [int(c.isupper()) for c in predictions]
    events_['pred_selected_high'] = (events_['pred_choice'] == events_['high_port']).astype(int)
    events_['prev_choice'] = events_['seq2_RL'].apply(lambda x: x[-1].upper()
                                                      if not pd.isna(x) else None)
    events_['pred_switch'] = (events_['pred_choice_str'] != events_['prev_choice']).astype(int)

    return events_
