import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap

from .graph_helper import (calculate_switch_probabilities, plot_probabilities,
                           plot_switch_probabilities)
import pandas as pd
import itertools

global rflr


def parse_files(behavior_filename, high_port_filename, clip_short_blocks=False):

    with open(behavior_filename, 'r') as behavior_file:
        behavior_data = behavior_file.read().replace('\n', '').replace(' ', '')
    with open(high_port_filename, 'r') as high_port_file:
        high_port_data = high_port_file.read().replace('\n', '').replace(' ', '')
    assert len(behavior_data) == len(high_port_data), (
        "Error: Behavior data and high port data have different lengths.")

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
        'choice': choices,
        'choice_str': choices_str,
        'reward': rewards,
        'selected_high': selected_high,
        'switch': switch,
        'transition': transitions,
        'high_port': high_ports,
    })

    # Calculate length of each block as num trial from previous transition. Prepend first block, and last
    # block is distance to end of sequence.
    block_lengths = [events.query('transition == 1')['trial_number'].values[0]]
    block_lengths.extend(events.query('transition == 1')['trial_number'].diff().values[1:].astype('int'))
    block_lengths.extend([len(events) - events.query('transition == 1')['trial_number'].values[-1]])

    # Store block lengths at transitions and fill backwards (so each trial can reference ultimate block length).
    events.loc[events.index[0], 'block_length'] = block_lengths[0]
    events.loc[events['transition']==1, 'block_length'] = block_lengths[1:]
    # events.loc[events.index[-1], 'block_length'] = block_lengths[-1]
    events['block_length'] = events['block_length'].ffill()

    # Counter for index position within each block. Forward and reverse (negative, from end of block backwards).
    block_positions = list(itertools.chain(*[np.arange(i) for i in block_lengths]))
    events['block_position'] = block_positions
    events['rev_block_position'] = events['block_position'] - events['block_length']

    # Unique ID for each block.
    events.loc[events['transition']==1, 'block_id'] = np.arange(1,len(block_lengths))
    events['block_id'] = events['block_id'].ffill()

    if clip_short_blocks: 
        # ID of short blocks, and blocks immediately following short blocks (not back to baseline).
        short_blocks = events.query('block_length < 20')['block_id'].unique()
        post_short_blocks = short_blocks + 1
        raise NotImplementedError
        # For reference:
        # sns.lineplot(data=events.query('block_position.between(0, 20) & ~block_id.isin(@post_short_blocks)'), x='block_position', y='switch', ax=ax)
        # sns.lineplot(data=events.query('rev_block_position.between(-10, 1) & block_length > 20'), x='rev_block_position', y='switch', ax=ax)

    return events


def calculate_probabilities(events):
    block_positions = list(range(-10, 21))

    # Initialize dictionaries to collect data
    high_reward_data = {pos: [] for pos in block_positions}
    switch_data = {pos: [] for pos in block_positions}

    for event in events:
        for pos in event['block_position']:
            if pos in block_positions:
                high_reward_data[pos].append(event['selected_high'])
                switch_data[pos].append(event['switch'])

    high_reward_prob = []
    high_reward_ci_lower = []
    high_reward_ci_upper = []
    switch_prob = []
    switch_ci_lower = []
    switch_ci_upper = []

    for pos in block_positions:
        if high_reward_data[pos]:
            data = np.array(high_reward_data[pos])
            prob = np.mean(data)
            res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic')
            ci_lower = res.confidence_interval.low
            ci_upper = res.confidence_interval.high
            high_reward_prob.append(prob)
            high_reward_ci_lower.append(ci_lower)
            high_reward_ci_upper.append(ci_upper)
        else:
            high_reward_prob.append(np.nan)
            high_reward_ci_lower.append(np.nan)
            high_reward_ci_upper.append(np.nan)

        if switch_data[pos]:
            data = np.array(switch_data[pos])
            prob = np.mean(data)
            res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic')
            ci_lower = res.confidence_interval.low
            ci_upper = res.confidence_interval.high
            switch_prob.append(prob)
            switch_ci_lower.append(ci_lower)
            switch_ci_upper.append(ci_upper)
        else:
            switch_prob.append(np.nan)
            switch_ci_lower.append(np.nan)
            switch_ci_upper.append(np.nan)

    return block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper


def main():
    prefix = ''
    ground_truth = True
    if ground_truth:
        prefix = 'rflr_1M'
    else:
        prefix = 'new_gen'

    # Define the file paths
    run_number = 0
    root = os.path.dirname(__file__)
    behavior_filename = os.path.join(os.path.dirname(root), 'data', f'2ABT_behavior_run_{run_number}v.txt')
    high_port_filename = os.path.join(os.path.dirname(root), 'data', f'2ABT_high_port_run_{run_number}v.txt')

    # Check if files exist
    if not os.path.exists(behavior_filename) or not os.path.exists(high_port_filename):
        print("Behavior file or high port file not found!")
    else:
        # Parse the files
        events = parse_files(behavior_filename, high_port_filename)
        if events is not None:
            # Calculate and print the percent of trials with a switch
            total_trials = len(events) - 1  # Exclude the first trial
            total_switches = sum(event['switch'] for event in events[1:])  # Exclude the first trial
            percent_switches = (total_switches / total_trials) * 100 if total_trials > 0 else 0

            print(f"Percent of trials with a switch: {percent_switches:.2f}%")

            # Uncomment to print the first 100 events for debugging
            # for event in events[:100]:
            #     print(event)

            # Calculate probabilities for block positions
            block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper = calculate_probabilities(events)
            print(high_reward_prob)
            print(switch_prob)

            # Plot the probabilities
            plot_probabilities(block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper, prefix)

            # Calculate switch probabilities
            sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts = calculate_switch_probabilities(events)

            # Plot the switch probabilities
            plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts, prefix)


# Main code
if __name__ == "__main__":
    main()
