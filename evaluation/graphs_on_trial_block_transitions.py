import os
# Add the project root directory to Python path
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

import utils.file_management as fm
from evaluation.graph_helper import (calc_bpos_behavior, plot_bpos_behavior,
                                     plot_conditional_switching)
from utils import parse_data

global rflr


def main(run=None, suffix: str = 'v'):

    # Get file paths
    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, suffix, subdir='seqs')
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, suffix, subdir='seqs')
    context_filename = fm.get_experiment_file("context_transitions_run_{}.txt", run, suffix, subdir='seqs')
    print(behavior_filename, '\n', high_port_filename, '\n', context_filename)

    assert fm.check_files_exist(behavior_filename, high_port_filename, context_filename)

    # Parse the files
    events = parse_data.parse_simulated_data(behavior_filename, high_port_filename, context_filename)

    if events is not None:
        # Calculate and print the percent of trials with a switch
        percent_switches = events['switch'].mean()*100
        print(f"Percent of trials with a switch: {percent_switches:.2f}%")

        # Calculate probabilities for block positions
        bpos = calc_bpos_behavior(events, add_cond_cols=['context', 'session'])
        plot_bpos_behavior(bpos, run, suffix=suffix, hue='context', subdir='agent_behavior')
        plot_conditional_switching(events, seq_length=2, run=run, suffix=suffix, subdir='agent_behavior')
        plot_conditional_switching(events, seq_length=3, run=run, suffix=suffix, subdir='agent_behavior')


# Main code
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)