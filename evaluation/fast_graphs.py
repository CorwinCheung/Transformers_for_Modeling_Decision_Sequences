import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_file(filename):
    """
    Parse the data file to extract relevant information and store it in a pandas DataFrame.

    Args:
    - filename (str): The file path of the data file.

    Returns:
    - df (DataFrame): A DataFrame containing all the parsed event data.
    """
    data = []
    last_choice = None
    current_state = 0  # Default starting state; 0 for left, 1 for right
    trial_number = 0
    block_position = 0
    swap_trials = []
    max_pre_swap = 10

    with open(filename, 'r') as file:
        tokens = []
        for line in file:
            tokens.extend(line.strip())

    # Process tokens
    for token in tokens:
        if token == 'O':  # Start state
            current_state = 1  # High-reward spout is on the right
            block_position = 0
            swap_trials.append(trial_number)
        elif token == 'S':  # Swap occurred
            current_state = 1 - current_state
            block_position = 0
            swap_trials.append(trial_number)
        elif token in 'LlRr':
            choice = 0 if token in 'Ll' else 1  # Left: 0, Right: 1
            choice_str = 'L' if choice == 0 else 'R'
            reward = 1 if token.isupper() else 0
            rewarded = bool(reward)
            selected_high = 1 if choice == current_state else 0
            switch = 1 if last_choice is not None and choice != last_choice else 0

            data.append({
                'trial_number': trial_number,
                'choice': choice,
                'choice_str': choice_str,
                'reward': reward,
                'rewarded': rewarded,
                'selected_high': selected_high,
                'switch': switch,
                'swap': 1 if trial_number in swap_trials else 0,
                'block_position': block_position
            })

            last_choice = choice
            trial_number += 1
            block_position += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Adjust block positions for trials before each swap
    for swap_trial in swap_trials:
        idx_range = range(max(0, swap_trial - max_pre_swap), swap_trial)
        df.loc[df['trial_number'].isin(idx_range), 'block_position'] = df.loc[df['trial_number'].isin(idx_range), 'trial_number'] - swap_trial

    return df

def calculate_probabilities(df):
    """
    Calculate probabilities for high-reward selection and switching around block transitions using vectorization.

    Args:
    - df (DataFrame): The parsed events from the data file.

    Returns:
    - block_positions (list): Block positions relative to swaps.
    - high_reward_prob (list): Probability of selecting the high-reward port.
    - switch_prob (list): Probability of switching sides.
    """
    block_positions = range(-10, 21)
    df_filtered = df[df['block_position'].isin(block_positions)]

    # Group by block_position
    grouped = df_filtered.groupby('block_position')
    high_reward_prob = grouped['selected_high'].mean()
    switch_prob = grouped['switch'].mean()

    # Align results with block_positions
    high_reward_prob = high_reward_prob.reindex(block_positions).tolist()
    switch_prob = switch_prob.reindex(block_positions).tolist()

    return block_positions, high_reward_prob, switch_prob

def plot_probabilities(block_positions, high_reward_prob, switch_prob, prefix=''):
    """
    Plot the probabilities of high-reward selection and switching relative to block positions.

    Args:
    - block_positions (list): Block positions relative to swaps.
    - high_reward_prob (list): Probability of selecting the high-reward port.
    - switch_prob (list): Probability of switching sides.
    - prefix (str): Prefix for the saved plot filenames.
    """
    # Probability of Selecting High-Reward Port
    plt.figure(figsize=(10, 5))
    plt.plot(block_positions, high_reward_prob, marker='o', color='blue')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Block Position")
    plt.ylabel("P(high port)")
    plt.title("Probability of Selecting High-Reward Port")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{prefix}selecting_high_reward_port.png')
    plt.close()

    # Probability of Switching
    plt.figure(figsize=(10, 5))
    plt.plot(block_positions, switch_prob, marker='o', color='red')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Block Position")
    plt.ylabel("P(switch)")
    plt.title("Probability of Switching")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{prefix}switch_probabilities.png')
    plt.close()

def map_sequence_to_pattern(seq):
    """
    Map a sequence of three actions to a pattern.

    Args:
    - seq (DataFrame): A DataFrame slice with three events.

    Returns:
    - pattern (str): The mapped pattern string.
    """
    action1, action2, action3 = seq.iloc[0], seq.iloc[1], seq.iloc[2]
    first_reward = 'A' if action1['rewarded'] else 'a'
    first_choice = action1['choice_str']

    def get_letter(action, first_choice):
        same_side = action['choice_str'] == first_choice
        reward_char = 'A' if action['rewarded'] else 'a'
        return reward_char if same_side else reward_char.upper() if action['rewarded'] else reward_char.lower()

    second_letter = get_letter(action2, first_choice)
    third_letter = get_letter(action3, first_choice)
    pattern = f"{first_reward}{second_letter}{third_letter}"
    return pattern

def calculate_switch_probabilities(df):
    """
    Calculate the probability of switching given the previous three actions.

    Args:
    - df (DataFrame): The DataFrame of events.

    Returns:
    - sorted_patterns (list): List of patterns sorted by ascending switch probability.
    - sorted_probabilities (list): Corresponding switch probabilities.
    - counts (list): Counts of each pattern.
    """
    patterns = []
    switches = []

    for i in range(len(df) - 3):
        seq = df.iloc[i:i+3]
        next_action = df.iloc[i+3]
        pattern = map_sequence_to_pattern(seq)
        switched = seq.iloc[2]['choice'] != next_action['choice']
        patterns.append(pattern)
        switches.append(int(switched))

    df_patterns = pd.DataFrame({'pattern': patterns, 'switched': switches})
    grouped = df_patterns.groupby('pattern')
    probabilities = grouped['switched'].mean()
    counts = grouped['switched'].count()

    # Sort patterns
    sorted_data = probabilities.sort_values().reset_index()
    sorted_counts = counts.loc[sorted_data['pattern']].reset_index(drop=True)

    sorted_patterns = sorted_data['pattern'].tolist()
    sorted_probabilities = sorted_data['switched'].tolist()
    counts = sorted_counts.tolist()

    return sorted_patterns, sorted_probabilities, counts

def plot_switch_probabilities(patterns, probabilities, counts, prefix=''):
    """
    Plot the switch probabilities as a bar chart.

    Args:
    - patterns (list): List of patterns.
    - probabilities (list): Corresponding switch probabilities.
    - counts (list): Counts of each pattern.
    - prefix (str): Prefix for the saved plot filenames.
    """
    plt.figure(figsize=(18, 6))
    bars = plt.bar(range(len(patterns)), probabilities, tick_label=patterns)

    # Annotate counts
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f'n={count}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('History')
    plt.ylabel('Probability of Switching')
    plt.title('Probability of Switching Given the Previous Three Actions')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{prefix}conditional_switching.png')
    plt.close()


filename = "../data/2ABT_logistic_run_4.txt"  # Adjust the filename as needed

# Parse the file and create DataFrame
df = parse_file(filename)

# Calculate and print the percent of trials with a switch
total_trials = len(df) - 1  # Exclude the first trial
total_switches = df['switch'].iloc[1:].sum()
percent_switches = (total_switches / total_trials) * 100
print(f"Percent of trials with a switch: {percent_switches:.2f}%")

# Calculate probabilities
block_positions, high_reward_prob, switch_prob = calculate_probabilities(df)

# Plot probabilities
plot_probabilities(block_positions, high_reward_prob, switch_prob, prefix="fast_rflr")

# Calculate switch probabilities based on patterns
patterns, probabilities, counts = calculate_switch_probabilities(df)

# Plot switch probabilities
plot_switch_probabilities(patterns, probabilities, counts, prefix="fast_rflr")