import matplotlib.pyplot as plt
import numpy as np
import os
from graph_helper import plot_probabilities, calculate_switch_probabilities, plot_switch_probabilities
from scipy.stats import bootstrap

global rflr

def parse_files(behavior_filename, high_port_filename, clip_short_blocks=False):
    events = []
    last_choice = None
    trial_number = 0
    transition_trials = []

    with open(behavior_filename, 'r') as behavior_file:
        behavior_data = behavior_file.read().replace('\n', '').replace(' ', '')
    with open(high_port_filename, 'r') as high_port_file:
        high_port_data = high_port_file.read().replace('\n', '').replace(' ', '')
    if len(behavior_data) != len(high_port_data):
        print("Error: Behavior data and high port data have different lengths.")
        return None
    previous_high_port = None
    for i in range(len(behavior_data)):
        token = behavior_data[i]
        current_high_port = int(high_port_data[i])
        if previous_high_port is not None and current_high_port != previous_high_port:
            transition_trials.append(trial_number)

        # Process the behavior token
        if token in 'LlRr':
            if token in 'Ll':
                choice = 0  # Left
                choice_str = 'L'
            elif token in 'Rr':
                choice = 1  # Right
                choice_str = 'R'

            reward = 1 if token.isupper() else 0  # Rewarded if uppercase
            rewarded = bool(reward)

            # Determine if agent selected the high-reward spout
            selected_high = 1 if choice == current_high_port else 0

            # Determine if agent switched sides from previous trial
            if last_choice is not None:
                switch = 1 if choice != last_choice else 0
            else:
                switch = 0  # No switch on first trial

            event = {
                'trial_number': trial_number,
                'choice': choice,
                'choice_str': choice_str,
                'reward': reward,
                'rewarded': rewarded,
                'selected_high': selected_high,
                'switch': switch,
                'transition': 1 if trial_number in transition_trials else 0,
                'block_position': [],
                'high_port': current_high_port,
            }
            events.append(event)
            last_choice = choice
            trial_number += 1
        else:
            print(f"Unexpected token '{token}' at trial {trial_number}")

        previous_high_port = current_high_port

    if clip_short_blocks: 
        block_starts = []
        block_ends = []

        # If there are transitions
        if transition_trials:
            # First block starts at 0
            block_starts.append(0)
            for t in transition_trials:
                block_ends.append(t - 1)
                block_starts.append(t)
            # The last block ends at the last trial
            block_ends.append(len(events) - 1)
        else:
            # No transitions, single block
            block_starts.append(0)
            block_ends.append(len(events) - 1)

        block_lengths = [block_ends[i] - block_starts[i] + 1 for i in range(len(block_starts))]
        # Filter out blocks with length < 15
        valid_blocks = [i for i, length in enumerate(block_lengths) if length >= 15]

        # Now, collect events in valid blocks
        valid_events = []
        for i in valid_blocks:
            start = block_starts[i]
            end = block_ends[i]
            for event in events[start:end + 1]:
                event['block_index'] = i
                valid_events.append(event)

        # Now update events
        events = valid_events

        # Reset trial numbers and recompute transitions
        last_high_port = None
        for idx, event in enumerate(events):
            event['trial_number'] = idx
            current_high_port = event['high_port']
            if last_high_port is not None and current_high_port != last_high_port:
                event['transition'] = 1
            else:
                event['transition'] = 0
            last_high_port = current_high_port

        # Now recompute transition trials
        transition_trials = [event['trial_number'] for event in events if event['transition'] == 1]
        for event in events:
            event['block_position'] = []
            for transition_trial in transition_trials:
                pos = event['trial_number'] - transition_trial
                if -10 <= pos <= 20:
                    event['block_position'].append(pos)
        return events
    else:
        last_transition_trial = 0
        for idx, event in enumerate(events):
            if event['transition'] == 1:
                last_transition_trial = event['trial_number']
            event['block_position'].append(event['trial_number'] - last_transition_trial)

        # Assign negative block positions for trials before transitions
        for transition_trial in transition_trials:
            # Go back up to 10 trials before the transition
            for i in range(1, min(11, transition_trial + 1)):
                idx = transition_trial - i
                if idx >= 0:
                    events[idx]['block_position'].append(-i)  # Append negative block position
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


# Main code

prefix = ''
ground_truth = True
if ground_truth:
    prefix = 'rflr_1M'
else:
    prefix = 'new_gen'

# Define the file paths
behavior_filename = "../data/2ABT_behavior_run_2.txt"
high_port_filename = "../data/2ABT_high_port_run_2.txt"

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
        # plot_probabilities(block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper, prefix)

        # Calculate switch probabilities
        sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts = calculate_switch_probabilities(events)

        # Plot the switch probabilities
        # plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts, prefix)