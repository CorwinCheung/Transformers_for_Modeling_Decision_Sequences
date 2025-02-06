import os
# Add the project root directory to Python path
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
import utils.file_management as fm
from utils.parse_data import parse_simulated_data

def analyze_data(behavior_filename, high_port_filename, session_filename):
    """
    Get basic stats about the simulated data from the behavior, high port
    and session files.
    """

    events = parse_simulated_data(behavior_filename, high_port_filename, session_filename)
    analysis = {k: {} for k in events['domain'].unique()}

    for grp, domain_data in events.groupby('domain'):
        analysis[grp]["switches_percentage"] = domain_data.switch.mean() * 100
        analysis[grp]["switches"] = domain_data.switch.sum()

        analysis[grp]["transitions"] = domain_data.transition.sum()
        analysis[grp]["transitions_percentage"] = domain_data.transition.mean() * 100

        analysis[grp]["rewarded_left"] = np.mean(domain_data['k0'] == 'L') * 100
        analysis[grp]["rewarded_right"] = np.mean(domain_data['k0'] == 'R') * 100
        analysis[grp]["unrewarded_left"] = np.mean(domain_data['k0'] == 'l') * 100
        analysis[grp]["unrewarded_right"] = np.mean(domain_data['k0'] == 'r') * 100
        analysis[grp]["total_trials"] = len(domain_data)
        analysis[grp]["selected_correct"] = np.sum(domain_data['selected_high'])
        analysis[grp]["selected_correct_percentage"] = np.mean(domain_data['selected_high']) * 100

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
    print(f"Percent of trials with a switch: {analysis['switches_percentage']:.2f}%\n")

def main(run=None):
    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, 'tr', subdir='seqs')
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, 'tr', subdir='seqs')
    session_filename = fm.get_experiment_file("session_transitions_run_{}.txt", run, 'tr', subdir='seqs')

    assert fm.check_files_exist(behavior_filename, high_port_filename, session_filename)
    print(f"Analyzing data from:\n {behavior_filename}\n {high_port_filename}")
    analysis = analyze_data(behavior_filename, high_port_filename, session_filename)
    if analysis:
        for domain, stats in analysis.items():
            print(f'\n\nAnalysis for Domain {domain}')
            print_table(stats)
            print_switches(stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)
