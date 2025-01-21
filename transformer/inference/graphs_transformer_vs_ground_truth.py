import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#so that I can import from a directory two levels up
from evaluation.graph_helper import plot_probabilities, calculate_switch_probabilities, plot_switch_probabilities
from scipy.stats import bootstrap

from utils.file_management import get_file_path


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
    block_positions = list(range(-10, 21))  # This range includes both negative and positive positions

    # Initialize lists to store probabilities and confidence intervals
    high_reward_prob = []
    high_reward_ci_lower = []
    high_reward_ci_upper = []
    switch_prob = []
    switch_ci_lower = []
    switch_ci_upper = []

    for pos in block_positions:
        selected_high = []
        switches = []

        # Gather data for the current block position
        for event in events:
            if pos in event['block_position']:
                selected_high.append(event['selected_high_prediction'])
                switches.append(event['switch'])

        # Calculate high-reward probabilities and bootstrap confidence intervals
        if selected_high:
            data = np.array(selected_high)
            prob = np.mean(data)
            high_reward_prob.append(prob)

            # Check if there are enough data points for bootstrapping
            if len(data) > 1:
                res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic')
                ci_lower = res.confidence_interval.low
                ci_upper = res.confidence_interval.high
            else:
                # If only one data point, set CI equal to the point estimate
                ci_lower = prob
                ci_upper = prob

            high_reward_ci_lower.append(ci_lower)
            high_reward_ci_upper.append(ci_upper)
        else:
            # No data for this position; append NaN
            high_reward_prob.append(np.nan)
            high_reward_ci_lower.append(np.nan)
            high_reward_ci_upper.append(np.nan)

        # Calculate switch probabilities and bootstrap confidence intervals
        if switches:
            data = np.array(switches)
            prob = np.mean(data)
            switch_prob.append(prob)

            if len(data) > 1:
                res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic')
                ci_lower = res.confidence_interval.low
                ci_upper = res.confidence_interval.high
            else:
                ci_lower = prob
                ci_upper = prob

            switch_ci_lower.append(ci_lower)
            switch_ci_upper.append(ci_upper)
        else:
            switch_prob.append(np.nan)
            switch_ci_lower.append(np.nan)
            switch_ci_upper.append(np.nan)

    # Return the block positions, probabilities, and confidence intervals
    return block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper

def main(run=None):
    # Files will automatically use latest run if run=None
    behavior_filename = get_file_path(f"behavior_run_{run}tr.txt", run)
    high_port_filename = get_file_path(f"high_port_run_{run}tr.txt", run)
    predictions_filename = get_file_path(f"pred_run_{run}.txt", run)

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
        block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper = calculate_probabilities(events)

        # Plot the probabilities
        plot_probabilities(block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper, f"run_{run}", "../")

        # Calculate switch probabilities
        sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts = calculate_switch_probabilities(events)

        # Plot the switch probabilities
        plot_switch_probabilities(sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts, f"run_{run}", "../")
