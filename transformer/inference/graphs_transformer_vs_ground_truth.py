import matplotlib.pyplot as plt
import numpy as np

global rflr

def parse_file(filename):
    """
    Parse the data file to extract ground truth events.

    Args:
    - filename (str): The file path of the ground truth data file.

    Returns:
    - events (list): A list of dictionaries with keys:
        - 'trial_number'
        - 'choice' (0 for left, 1 for right)
        - 'choice_str' ('L' or 'R')
        - 'reward' (1 for rewarded, 0 for unrewarded)
        - 'rewarded' (True or False)
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
                    last_swap_trial = trial_number  # Treat 'O' as a swap at the beginning
                    swap_trials.append(trial_number)
                elif token == 'S':  # Swap occurred
                    current_state = 1 - current_state  # Swap high-reward spout position
                    last_swap_trial = trial_number  # Set swap point at current trial
                    swap_trials.append(trial_number)
                elif token in 'LlRr':  # Process the choice
                    if token in 'Ll':
                        choice = 0  # Left
                        choice_str = 'L'
                    elif token in 'Rr':
                        choice = 1  # Right
                        choice_str = 'R'

                    reward = 1 if token.isupper() else 0  # Rewarded if uppercase
                    rewarded = bool(reward)

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
                        'choice_str': choice_str,
                        'reward': reward,
                        'rewarded': rewarded,
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
            if idx >= 0:
                events[idx]['block_position'] = -i

    return events

def read_predictions(filename):
    """
    Read the prediction file and return the prediction sequence.

    Args:
    - filename (str): The file path of the prediction data file.

    Returns:
    - predictions (str): The prediction sequence as a string.
    """
    with open(filename, 'r') as f:
        predictions = f.read().replace('\n', '')
        predictions = predictions.replace('S', '')
        predictions = predictions.replace('O', '')
    return predictions

def align_events_with_predictions(events, predictions):
    """
    Align the ground truth events with the predictions.

    Args:
    - events (list): The list of ground truth events.
    - predictions (str): The prediction sequence.

    Returns:
    - events (list): The updated events with predictions and adjusted switches.
    """
    # Ensure the sequences are of the same length
    min_length = min(len(events), len(predictions))
    events = events[:min_length]
    predictions = predictions[:min_length]

    # Initialize the last ground truth choice
    last_ground_truth_choice = None

    for i in range(len(events)):
        event = events[i]
        pred_char = predictions[i]

        # Map prediction character to choice
        if pred_char in 'Ll':
            pred_choice = 0  # Left
            pred_choice_str = 'L'
        elif pred_char in 'Rr':
            pred_choice = 1  # Right
            pred_choice_str = 'R'
        else:
            pred_choice = None
            pred_choice_str = None

        event['prediction_choice'] = pred_choice
        event['prediction_choice_str'] = pred_choice_str

        # Calculate switch based on ground truth choice at previous trial
        if i > 0 and last_ground_truth_choice is not None:
            switch = 1 if pred_choice != last_ground_truth_choice else 0
        else:
            switch = 0  # No switch on first trial

        event['switch'] = switch  # Update the switch definition

        last_ground_truth_choice = event['choice']  # Update for next iteration

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
    Plot the probabilities of high-reward selection and switching relative to block positions
    as two separate plots, each with their own y-axis limits.

    Args:
    - block_positions (list): Block positions relative to swaps.
    - high_reward_prob (list): Probability of selecting the high-reward port.
    - switch_prob (list): Probability of switching sides.
    """
    # Plot P(high port)
    plt.figure(figsize=(10, 5))
    plt.plot(block_positions, high_reward_prob, label="P(high port)", marker='o', color='blue')
    plt.axvline(0, color='black', linestyle='--', label="Block Transition")
    plt.xlabel("Block Position")
    plt.ylabel("P(high port)")
    plt.title("Probability of Selecting High-Reward Port")
    plt.legend()
    plt.ylim(0, 1)  # Adjust y-axis limits as needed
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../graphs/{rflr}_G_selecting high-reward port.png')
    # plt.show()
    # Plot P(switch)
    plt.figure(figsize=(10, 5))
    plt.plot(block_positions, switch_prob, label="P(switch)", marker='o', color='red')
    plt.axvline(0, color='black', linestyle='--', label="Block Transition")
    plt.xlabel("Block Position")
    plt.ylabel("P(switch)")
    plt.title("Probability of Switching")
    plt.legend()
    plt.ylim(0, max(switch_prob) * 1.1)  # Adjust y-axis limits based on data
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../graphs/{rflr}_G_switch probabilities.png')
    # plt.show()

def map_sequence_to_pattern(seq):
    """
    Map a sequence of three actions to a pattern as per the specified rules.

    Args:
    - seq (list): A list of three event dictionaries.

    Returns:
    - pattern (str): The mapped pattern string.
    """
    action1, action2, action3 = seq

    # First action: 'A' if rewarded, 'a' if unrewarded
    first_reward = 'A' if action1['rewarded'] else 'a'
    first_choice = action1['prediction_choice_str']  # Use prediction choice

    # Second action
    second_same_side = action2['prediction_choice_str'] == first_choice
    if second_same_side:
        second_letter = 'A' if action2['rewarded'] else 'a'
    else:
        second_letter = 'B' if action2['rewarded'] else 'b'

    # Third action
    third_same_side = action3['prediction_choice_str'] == first_choice
    if third_same_side:
        third_letter = 'A' if action3['rewarded'] else 'a'
    else:
        third_letter = 'B' if action3['rewarded'] else 'b'

    # Combine letters to form the pattern
    pattern = f"{first_reward}{second_letter}{third_letter}"

    return pattern

def calculate_switch_probabilities(events):
    """
    Calculate the probability of switching given the previous three actions.

    Args:
    - events (list): The list of events.

    Returns:
    - sorted_patterns (list): List of patterns sorted by ascending switch probability.
    - sorted_probabilities (list): Corresponding switch probabilities.
    - counts (list): Counts of each pattern.
    """
    pattern_data = {}

    for i in range(3, len(events)):
        # Previous three actions
        seq = events[i-3:i]
        # Next action
        next_action = events[i]

        # Map the sequence to a pattern
        pattern = map_sequence_to_pattern(seq)

        # Determine if agent switched on the next trial (based on ground truth)
        switched = seq[2]['prediction_choice'] != next_action['prediction_choice']

        # Update counts
        if pattern not in pattern_data:
            pattern_data[pattern] = {'switches': 0, 'total': 0}
        pattern_data[pattern]['total'] += 1
        if switched:
            pattern_data[pattern]['switches'] += 1

    # Calculate probabilities and prepare for sorting
    patterns = []
    probabilities = []
    counts = []
    for pattern, data in pattern_data.items():
        total = data['total']
        switches = data['switches']
        prob = switches / total if total > 0 else 0
        patterns.append(pattern)
        probabilities.append(prob)
        counts.append(total)

    # Sort patterns by ascending switch probability
    sorted_indices = np.argsort(probabilities)
    sorted_patterns = [patterns[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    return sorted_patterns, sorted_probabilities, sorted_counts

def plot_switch_probabilities(patterns, probabilities, counts):
    """
    Plot the switch probabilities as a bar chart.

    Args:
    - patterns (list): List of patterns.
    - probabilities (list): Corresponding switch probabilities.
    - counts (list): Counts of each pattern.
    """
    # Create the bar chart
    plt.figure(figsize=(18, 6))
    bars = plt.bar(range(len(patterns)), probabilities, tick_label=patterns)

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'n={count}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('History')
    plt.ylabel('Probability of Switching')
    plt.title('Probability of Switching Given the Previous Three Actions')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'../graphs/{rflr}_F_conditional switching.png')
    # plt.show()

# Main code

rflr = ''
ground_truth = False
if ground_truth:
    rflr = 'rflr_'
else:
    rflr = 'model_92k'

# Define the file paths
ground_truth_file = "../data/2ABT_logistic_run_4.txt"
predictions_file = "../transformer/inference/Preds_for_4_with_model_92k.txt"

# Parse the ground truth events
events = parse_file(ground_truth_file)

# Read predictions
predictions = read_predictions(predictions_file)

# Align events with predictions and adjust switches
events = align_events_with_predictions(events, predictions)

# For debugging: print a sample of events
# for event in events[:30]:
#     print(event)

# Calculate and print the percent of trials with a switch
total_trials = len(events) - 1  # Exclude the first trial
total_switches = sum(event['switch'] for event in events[1:])  # Exclude the first trial
percent_switches = (total_switches / total_trials) * 100

print(f"Percent of trials with a switch: {percent_switches:.2f}%")

# Calculate probabilities for block positions
block_positions, high_reward_prob, switch_prob = calculate_probabilities(events)

# Plot the probabilities
plot_probabilities(block_positions, high_reward_prob, switch_prob)

# Calculate switch probabilities
sorted_patterns, sorted_probabilities, sorted_counts = calculate_switch_probabilities(events)

# Plot the switch probabilities
plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_counts)
