import matplotlib.pyplot as plt
import numpy as np

def parse_file(filename):
    """
    Parse the data file to extract relevant information: choices, rewards, selected_high, block_position, and switches.

    Args:
    - filename (str): The file path of the data file.

    Returns:
    - events (list): A list of dictionaries with keys:
        - 'trial_number'
        - 'choice' (0 for left, 1 for right)
        - 'reward' (1 for rewarded, 0 for unrewarded)
        - 'selected_high' (1 if high-reward spout selected, 0 otherwise)
        - 'switch' (1 if switched from last choice, 0 otherwise)
        - 'swap' (1 if swap occurred on this trial, 0 otherwise)
        - 'block_position' (position relative to last swap)
    """
    events = []
    last_choice = None
    current_state = 0  # Default starting state; 0 for high-reward on left, 1 for high-reward on right
    trial_number = 0
    last_swap_trial = 0  # Initialize to 0
    swap_trials = []  # List to store the trial numbers where swaps occurred

    with open(filename, 'r') as file:
        for line in file:
            for token in line.strip():
                if token == 'O':  # Starts on right
                    current_state = 1  # High-reward spout is on the right
                elif token == 'S':  # Swap occurred
                    current_state = 1 - current_state  # Swap high-reward spout position
                    last_swap_trial = trial_number  # Set swap point at current trial
                    swap_trials.append(trial_number)
                elif token in 'LlRr':  # Process the choice
                    if token in 'Ll':
                        choice = 0  # Left
                    elif token in 'Rr':
                        choice = 1  # Right

                    reward = 1 if token.isupper() else 0  # Rewarded if uppercase

                    # Determine if agent selected the high-reward spout
                    selected_high = 1 if choice == current_state else 0

                    # Determine if agent switched sides from previous trial
                    if last_choice is not None:
                        switch = 1 if choice != last_choice else 0
                    else:
                        switch = 0  # No switch on first trial

                    # Calculate block_position relative to last swap
                    block_position = trial_number - last_swap_trial

                    # Record the event
                    event = {
                        'trial_number': trial_number,
                        'choice': choice,
                        'reward': reward,
                        'selected_high': selected_high,
                        'switch': switch,
                        'swap': 1 if trial_number in swap_trials else 0,
                        'block_position': block_position
                    }

                    events.append(event)

                    last_choice = choice
                    trial_number += 1
                else:
                    continue  # Ignore other tokens

    # Adjust block positions for trials before each swap
    for swap_trial in swap_trials:
        # Go back up to 10 trials before the swap
        for i in range(1, min(11, swap_trial + 1)):
            idx = swap_trial - i
            events[idx]['block_position'] = -i

    return events

def calculate_probabilities(events):
    """
    Calculate probabilities for high-reward selection and switching around block transitions.

    Args:
    - events (list): The parsed events from the data file.

    Returns:
    - block_positions (list): Block positions relative to swaps.
    - high_reward_prob (list): Probability of selecting the high-reward port.
    - switch_prob (list): Probability of switching sides (left to right or vice versa).
    """
    block_positions = list(range(-10, 21))
    high_reward_prob = []
    switch_prob = []

    for pos in block_positions:
        selected_high = [event['selected_high'] for event in events if event['block_position'] == pos]
        switches = [event['switch'] for event in events if event['block_position'] == pos]

        if selected_high:
            high_reward_prob.append(np.mean(selected_high))
        else:
            high_reward_prob.append(np.nan)  # Use NaN for positions with no data

        if switches:
            switch_prob.append(np.mean(switches))
        else:
            switch_prob.append(np.nan)  # Use NaN for positions with no data

    return block_positions, high_reward_prob, switch_prob

def plot_probabilities(block_positions, high_reward_prob, switch_prob):
    """
    Plot the probabilities of high-reward selection and switching relative to block positions.

    Args:
    - block_positions (list): Block positions relative to swaps.
    - high_reward_prob (list): Probability of selecting the high-reward port.
    - switch_prob (list): Probability of switching sides.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot P(high port)
    axes[0].plot(block_positions, high_reward_prob, label="P(high port)", marker='o')
    axes[0].axvline(0, color='black', linestyle='--', label="Block Transition")
    axes[0].set_xlabel("Block Position")
    axes[0].set_ylabel("P(high port)")
    axes[0].set_title("Probability of Selecting High-Reward Port")
    axes[0].legend()
    axes[0].set_ylim(0, 1)  # Set y-axis limits to [0,1]

    # Plot P(switch)
    axes[1].plot(block_positions, switch_prob, label="P(switch)", color='red', marker='o')
    axes[1].axvline(0, color='black', linestyle='--', label="Block Transition")
    axes[1].set_xlabel("Block Position")
    axes[1].set_ylabel("P(switch)")
    axes[1].set_title("Probability of Switching")
    axes[1].legend()
    axes[1].set_ylim(0, 1)  # Set y-axis limits to [0,1]

    plt.tight_layout()
    plt.show()

# Define the file path
filename = "../data/2ABT_logistic_run_2.txt"

# Parse the file
events = parse_file(filename)

# Calculate probabilities
block_positions, high_reward_prob, switch_prob = calculate_probabilities(events)

# Plot the results
plot_probabilities(block_positions, high_reward_prob, switch_prob)
