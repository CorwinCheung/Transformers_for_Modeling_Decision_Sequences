import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import get_experiment_file

sys.path.append('/Users/celiaberon/GitHub/behavior-helpers')
from bh.visualization import plot_trials


def plot_probabilities(events, run):
    bpos = plot_trials.calc_bpos_probs(events)
    fig, axs = plot_trials.plot_bpos_behavior(bpos)
    [ax.set(xlim=(-10, 20)) for ax in axs]
    axs[1].set(ylim=(0, 0.2))
    bpos_filename = get_experiment_file('bpos_behavior_{}.png', run, 'v')
    fig.savefig(bpos_filename, bbox_inches='tight')


# def plot_probabilities(block_positions, high_reward_prob, high_reward_ci_lower, high_reward_ci_upper, switch_prob, switch_ci_lower, switch_ci_upper, prefix, directory_escape=""):
#     # Plot P(high port)
#     plt.figure(figsize=(10, 5))
#     plt.plot(block_positions, high_reward_prob, label="P(high port)", marker='o', color='blue')
#     plt.fill_between(block_positions, high_reward_ci_lower, high_reward_ci_upper, color='blue', alpha=0.2)
#     plt.axvline(0, color='black', linestyle='--', label="Block Transition")
#     plt.xlabel("Block Position")
#     plt.ylabel("P(high port)")
#     plt.title("Probability of Selecting High-Reward Port with 95% CI")
#     plt.legend()
#     plt.ylim(0, 1)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'../{directory_escape}graphs/{prefix}_G_selecting_high_reward_port.png')

#     # Plot P(switch)
#     plt.figure(figsize=(10, 5))
#     plt.plot(block_positions, switch_prob, label="P(switch)", marker='o', color='blue')
#     plt.fill_between(block_positions, switch_ci_lower, switch_ci_upper, color='blue', alpha=0.2)
#     plt.axvline(0, color='black', linestyle='--', label="Block Transition")
#     plt.xlabel("Block Position")
#     plt.ylabel("P(switch)")
#     plt.title("Probability of Switching with 95% CI")
#     plt.legend()
#     plt.ylim(0, 1.1 * max(switch_prob))
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'../{directory_escape}graphs/{prefix}_G_switch_probabilities.png')

def map_sequence_to_pattern(seq):
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
    pattern_data = {}

    for i in range(3, len(events)):
        # Previous three actions
        seq = events[i-3:i]
        # Next action
        next_action = events[i]

        # Map the sequence to a pattern
        pattern = map_sequence_to_pattern(seq)

        # Determine if agent switched on the next trial
        switched = next_action['switch']

        # Update counts
        if pattern not in pattern_data:
            pattern_data[pattern] = []
        pattern_data[pattern].append(switched)

    patterns = []
    probabilities = []
    counts = []
    ci_lower = []
    ci_upper = []

    for pattern, data in pattern_data.items():
        data_array = np.array(data)
        total = len(data_array)
        switches = np.sum(data_array)
        prob = switches / total if total > 0 else 0
        res = bootstrap((data_array,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic')
        patterns.append(pattern)
        probabilities.append(prob)
        counts.append(total)
        ci_lower.append(res.confidence_interval.low)
        ci_upper.append(res.confidence_interval.high)

    # Sort patterns alphabetically
    sorted_indices = np.argsort(patterns)
    sorted_patterns = [patterns[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_ci_lower = [ci_lower[i] for i in sorted_indices]
    sorted_ci_upper = [ci_upper[i] for i in sorted_indices]

    return sorted_patterns, sorted_probabilities, sorted_ci_lower, sorted_ci_upper, sorted_counts


def plot_switch_probabilities(patterns, probabilities, ci_lower, ci_upper, counts, prefix, directory_escape=""):
    # Calculate error bars (asymmetric errors)
    lower_errors = [probabilities[i] - ci_lower[i] for i in range(len(probabilities))]
    upper_errors = [ci_upper[i] - probabilities[i] for i in range(len(probabilities))]
    errors = [lower_errors, upper_errors]

    # Create the bar chart
    plt.figure(figsize=(18, 6))
    bars = plt.bar(range(len(patterns)), probabilities, yerr=errors, tick_label=patterns, capsize=5)

    # Annotate bars with counts, positioning the text above the error bars
    for idx, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        upper_error = upper_errors[idx]
        # Set the text position above the upper error bar
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + upper_error + 0.02,  # Adjust 0.02 as needed for spacing
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.xlabel('History')
    plt.ylabel('Probability of Switching')
    plt.title('Probability of Switching Given the Previous Three Actions with 95% CI')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'../{directory_escape}graphs/{prefix}_F_conditional_switching.png')
