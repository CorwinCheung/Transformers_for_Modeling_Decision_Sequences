import os
# Add the project root directory to Python path
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import get_experiment_file, read_sequence


def analyze_data(behavior_filename, high_port_filename):

    behavior_data = read_sequence(behavior_filename)
    high_port_data = read_sequence(high_port_filename)

    if len(behavior_data) != len(high_port_data):
        print("Error: Data lengths do not match.")
        return None

    high_port_data = [int(hp) for hp in high_port_data]
    transitions = np.abs(np.diff(high_port_data))

    choice_data = [1 if choice.upper() == 'R' else 0
                   for choice in behavior_data]
    selected_correct = [hp == choice
                        for hp, choice in zip(high_port_data, choice_data)]
    switch = np.abs(np.diff(choice_data))
    analysis = {"transitions": transitions.sum(),
                "transitions_percentage": transitions.mean() * 100,
                "total_trials": len(behavior_data),
                "selected_correct": np.sum(selected_correct),
                "selected_correct_percentage": np.mean(selected_correct) * 100,
                "switches": switch.sum(),
                "switches_percentage": switch.mean() * 100}
    
    percent_factor = 100 / analysis["total_trials"]
    analysis["rewarded_left"] = behavior_data.count('L') * percent_factor
    analysis["rewarded_right"] = behavior_data.count('R') * percent_factor
    analysis["unrewarded_left"] = behavior_data.count('l') * percent_factor
    analysis["unrewarded_right"] = behavior_data.count('r') * percent_factor

    return analysis

def print_table(analysis):
    print(f"{'':<20} {'Left':>10} {'Right':>10}")
    print("="*40)
    print(f"{'Rewarded (%)':<20} {analysis['rewarded_left']:>10.2f}% {analysis['rewarded_right']:>10.2f}%")
    print(f"{'Unrewarded (%)':<20} {analysis['unrewarded_left']:>10.2f}% {analysis['unrewarded_right']:>10.2f}%")
    print("="*40)
    print(f"{'Total Trials:':<20} {analysis['total_trials']:>10,}")
    print(f"{'Number of Transitions:':<20} {analysis['transitions']:>10,} ({analysis['transitions_percentage']:.2f}% of total trials)")
    print(f"{'Selected Correct (%):':<20} {analysis['selected_correct_percentage']:>10.2f}%")

def print_switches(analysis):

    print(f"Total trials: {analysis['total_trials']:,}")
    print(f"Total switches: {analysis['switches']:,}")
    print(f"Percent of trials with a switch: {analysis['switches_percentage']:.2f}%")

def main(run=None):
    behavior_filename = get_experiment_file("behavior_run_{}.txt", run, 'tr')
    high_port_filename = get_experiment_file("high_port_run_{}.txt", run, 'tr')

    if os.path.exists(behavior_filename) and os.path.exists(high_port_filename):
        print(f"Analyzing data from:\n {behavior_filename}\n {high_port_filename}")
        analysis = analyze_data(behavior_filename, high_port_filename)
        if analysis:
            print_table(analysis)
            print_switches(analysis)
    else:
        print(f"Files {behavior_filename} or {high_port_filename} not found!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)
