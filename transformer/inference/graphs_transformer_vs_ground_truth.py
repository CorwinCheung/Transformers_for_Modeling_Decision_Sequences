import matplotlib.pyplot as plt
import numpy as np
import os

global rflr

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

    last_transition_trial = 0
    transition_trials = []
    # Process each trial
    for i in range(len(behavior_data)):
        token = behavior_data[i]
        current_high_port = int(high_port_data[i])  # 0 for left, 1 for right

        # Determine if a transition occurred
        if previous_high_port is not None and current_high_port != previous_high_port:
            # Transition occurred
            last_transition_trial = trial_number
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
                'block_position': [trial_number - last_transition_trial],  # Will update later
                'current_state': current_high_port  # Add current_state
            }

            events.append(event)

            last_choice = choice
            trial_number += 1
        else:
            print(f"Unexpected token '{token}' at trial {trial_number}")

        # Update previous high port
        previous_high_port = current_high_port

    # Assign negative block positions for trials before transitions
    for transition_trial in transition_trials:
        # Go back up to 10 trials before the transition
        for i in range(1, min(11, transition_trial + 1)):
            idx = transition_trial - i
            if idx >= 0:
                events[idx]['block_position'].append(-i)  # Assign negative block position

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
        predictions = f.read().replace('\n', '').replace(' ', '')
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

    # Initialize the last prediction choice
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

        event['prediction_choice'] = pred_choice
        event['prediction_choice_str'] = pred_choice_str
        event['selected_high_prediction'] = 1 if pred_choice == event['current_state'] else 0

        # Calculate switch based on previous prediction
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
    - block_positions (list): Block positions relative to transitions.
    - high_reward_prob (list): Probability of selecting the high-reward port.
    - switch_prob (list): Probability of switching sides.
    """
    block_positions = list(range(-10, 21))  # This range includes both negative and positive
    high_reward_prob = []
    switch_prob = []

    for pos in block_positions:
        selected_high = []
        switches = []

        # Gather probabilities for both positive and negative positions
        for event in events:
            if pos in event['block_position']:  # If the position (either positive or negative) is in the list
                selected_high.append(event['selected_high_prediction'])
                switches.append(event['switch'])

        # Calculate probabilities for each block position
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
    - block_positions (list): Block positions relative to transition.
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
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../../graphs/{rflr}_G_selecting_high_reward_port.png')

    # Plot P(switch)
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
    plt.savefig(f'../../graphs/{rflr}_G_switch_probabilities.png')

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
    first_choice = action1['choice_str']

    # Second action
    second_same_side = action2['choice_str'] == first_choice
    if second_same_side:
        second_letter = 'A' if action2['rewarded'] else 'a'
    else:
        second_letter = 'B' if action2['rewarded'] else 'b'

    # Third action
    third_same_side = action3['choice_str'] == first_choice
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

        # Determine if agent switched on the next trial (based on predictions)
        switched = 1 if next_action['prediction_choice'] != seq[-1]['prediction_choice'] else 0

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

    # Sort patterns alphabetically
    sorted_indices = np.argsort(patterns)
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
    plt.savefig(f'../../graphs/{rflr}_F_conditional_switching.png')

# Main code

rflr = 'model_100k'

# Define the file paths
behavior_filename = "../../data/2ABT_behavior_run_3.txt"
high_port_filename = "../../data/2ABT_high_port_run_3.txt"
predictions_filename = "Preds_for_3_with_model_seen92M.txt"

# Check if files exist
if not os.path.exists(behavior_filename) or not os.path.exists(high_port_filename) or not os.path.exists(predictions_filename):
    print("One or more files not found!")
else:
    # Parse the ground truth events
    events = parse_files(behavior_filename, high_port_filename)

    # Read predictions
    predictions = read_predictions(predictions_filename)
    print(f"Number of events: {len(events)}")
    print(f"Number of predictions: {len(predictions)}")

    # Align events with predictions and adjust switches
    events = align_events_with_predictions(events, predictions)

    # Calculate and print the percent of trials with a switch
    total_trials = len(events) - 1  # Exclude the first trial
    total_switches = sum(event['switch'] for event in events[1:])  # Exclude the first trial
    percent_switches = (total_switches / total_trials) * 100 if total_trials > 0 else 0

    print(f"Percent of trials with a switch: {percent_switches:.2f}%")

    # Calculate probabilities for block positions
    block_positions, high_reward_prob, switch_prob = calculate_probabilities(events)

    # Plot the probabilities
    plot_probabilities(block_positions, high_reward_prob, switch_prob)

    # Calculate switch probabilities
    sorted_patterns, sorted_probabilities, sorted_counts = calculate_switch_probabilities(events)

    # Plot the switch probabilities
    plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_counts)
