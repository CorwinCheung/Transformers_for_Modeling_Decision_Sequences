import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def parse_files(behavior_filename, high_port_filename):
    """
    Parse the behavior and high port files to extract events.

    Args:
    - behavior_filename (str): The file path of the behavior data file.
    - high_port_filename (str): The file path of the high port data file.

    Returns:
    - events (list): A list of dictionaries containing event data.
    """
    events = []
    last_choice = None
    trial_number = 0
    transition_trials = []  # List to store the trial numbers where transitions occurred

    # Read behavior data
    with open(behavior_filename, 'r') as behavior_file:
        behavior_data = behavior_file.read().replace('\n', '').replace(' ', '')

    # Read high port data
    with open(high_port_filename, 'r') as high_port_file:
        high_port_data = high_port_file.read().replace('\n', '').replace(' ', '')

    # Ensure both data have the same length
    if len(behavior_data) != len(high_port_data):
        print("Error: Behavior data and high port data have different lengths.")
        return None

    # Initialize previous high port
    previous_high_port = None

    # Process each trial
    for i in range(len(behavior_data)):
        token = behavior_data[i]
        current_high_port = int(high_port_data[i])  # 0 for left, 1 for right

        # Determine if a transition occurred
        if previous_high_port is not None and current_high_port != previous_high_port:
            # Transition occurred
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

            # Record the event
            event = {
                'trial_number': trial_number,
                'choice': choice,
                'choice_str': choice_str,
                'reward': reward,
                'rewarded': rewarded,
                'selected_high': selected_high,
                'switch': switch,
                'transition': 1 if trial_number in transition_trials else 0,
                'block_position': [],  # Will update later
                'high_port': current_high_port,
            }

            events.append(event)

            last_choice = choice
            trial_number += 1
        else:
            print(f"Unexpected token '{token}' at trial {trial_number}")

        # Update previous high port
        previous_high_port = current_high_port

    # Build blocks
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

    # Now compute block lengths
    block_lengths = [block_ends[i] - block_starts[i] + 1 for i in range(len(block_starts))]

    # Now filter out blocks with length < 15
    valid_blocks = [i for i, length in enumerate(block_lengths) if length >= 15]

    # Now, collect events in valid blocks
    valid_events = []
    for i in valid_blocks:
        start = block_starts[i]
        end = block_ends[i]
        for event in events[start:end + 1]:
            # Assign block index to event
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

    # Now adjust block positions
    for event in events:
        event['block_position'] = []
        for transition_trial in transition_trials:
            pos = event['trial_number'] - transition_trial
            # Only consider positions within the desired range
            if -10 <= pos <= 20:
                event['block_position'].append(pos)

    return events

def calculate_switch_probabilities(events):
    block_positions = list(range(-10,21))

    switch_data = {pos: [] for pos in block_positions}

    for event in events:
        for pos in event['block_position']:
            if pos in block_positions:
                switch_data[pos].append(event['switch'])
    switch_prob = []
    for pos in block_positions:
        switch_prob.append(np.mean(switch_data[pos]))
    return block_positions, switch_prob

def plot_switch_probabilities(block_positions, switch_prob):
    """
    Plot the switch probabilities as a line chart.

    Args:
    - block_positions (list): List of block positions.
    - switch_prob (list): Corresponding switch probabilities.
    """
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(block_positions, switch_prob, label="P(switch)", marker='o', color='blue')
    plt.axvline(0, color='black', linestyle='--', label="Block Transition")
    plt.xlabel("Block Position")
    plt.ylabel("P(switch)")
    plt.title("Probability of Switching")
    plt.legend()
    plt.ylim(0, max(switch_prob) * 1.1) 
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../graphs/small_test_G_switch_probabilities.png')
    # plt.show()

# Define the file paths
behavior_filename = "../data/2ABT_behavior_run_3.txt"
high_port_filename = "../data/2ABT_high_port_run_3.txt"

# behavior_filename = "../data/test.txt"
# high_port_filename = "../data/test_high_port.txt"

# Check if files exist
if not os.path.exists(behavior_filename) or not os.path.exists(high_port_filename):
    print("Behavior file or high port file not found!")
else:
    # Parse the files
    events = parse_files(behavior_filename, high_port_filename)
    # for e in events:
    #     print(e)

    if events is not None:
        # Calculate and print the percent of trials with a switch
        total_trials = len(events) - 1  # Exclude the first trial
        total_switches = sum(event['switch'] for event in events[1:])  # Exclude the first trial
        percent_switches = (total_switches / total_trials) * 100 if total_trials > 0 else 0

        print(f"Percent of trials with a switch: {percent_switches:.2f}%")

        # Calculate probabilities for block positions
        block_positions, switch_prob = calculate_switch_probabilities(events)

        # Plot the probabilities
        plot_switch_probabilities(block_positions, switch_prob)
