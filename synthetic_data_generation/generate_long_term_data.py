import configparser
import os
import random
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.file_management as fm
from synthetic_data_generation.generate_data import (
    generate_data, initialize_logger, write_session_transitions, generate_session
)
from environment import Original_2ABT_Spouts
from agent import RFLR_mouse

def modify_sequence_with_pattern(behavior_data, pattern_length):
    """Modify sequence to insert 'R' at every pattern_length steps."""
    modified_data = behavior_data.copy()
    for i in range(pattern_length-1, len(modified_data), pattern_length):
        modified_data[i] = 'R'  # Force 'R' at every pattern_length position
    return modified_data

def generate_patterned_data(num_steps, task_params, pattern_length=None):
    """Generate data with a forced pattern every pattern_length steps."""
    # First generate normal behavior
    behavior_data, high_port_data, session_transitions = generate_data(num_steps, task_params)
    
    # If pattern_length is specified, modify the sequence
    if pattern_length is not None:
        behavior_data = modify_sequence_with_pattern(behavior_data, pattern_length)
    
    return behavior_data, high_port_data, session_transitions

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--num_steps_train', type=int, default=100000)
    parser.add_argument('--num_steps_val', type=int, default=100000)
    parser.add_argument('--train_pattern', type=int, default=11,
                      help='Length of pattern to insert in training data')
    parser.add_argument('--val_pattern', type=int, default=13,
                      help='Length of pattern to insert in validation data')
    parser.add_argument('--no_overwrite', action='store_false', default=True,
                      help='Pass flag to prevent overwriting existing data')
    parser.add_argument('--config_file', type=str, default='three_domains.ini',
                      help='Configuration file for domains')
    return parser.parse_args()

def load_param_sets(config_file):
    """Load parameters from config file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    config.read(config_path)
    
    task_params = {}
    for section in config.sections():
        task_params[section] = {
            'environment': eval(config[section]['environment']),
            'agent': eval(config[section]['agent'])
        }
    return task_params

def main():
    args = parse_args()
    
    # Get next run number
    next_run = args.run or (fm.get_latest_run() + 1)
    
    # Initialize logger
    initialize_logger(next_run)
    
    # Load domain configurations (use domain B for consistency)
    task_params = {'B': load_param_sets(args.config_file)['B']}
    
    # Generate training data with 11-step pattern
    run_dir = fm.ensure_run_dir(next_run, subdir='seqs')
    
    # Training data
    behavior_filename_tr = fm.get_experiment_file("behavior_run_{}.txt", next_run, "tr", subdir='seqs')
    high_port_filename_tr = fm.get_experiment_file("high_port_run_{}.txt", next_run, "tr", subdir='seqs')
    sessions_filename_tr = fm.get_experiment_file("session_transitions_run_{}.txt", next_run, "tr", subdir='seqs')
    
    if fm.check_files_exist(behavior_filename_tr, high_port_filename_tr, sessions_filename_tr, verbose=False) and (not args.no_overwrite):
        print(f"Training files already exist for run_{next_run}, skipping data generation")
    else:
        behavior_data_tr, high_port_data_tr, session_transitions_tr = generate_patterned_data(
            args.num_steps_train, task_params, pattern_length=args.train_pattern)
        
        fm.write_sequence(behavior_filename_tr, behavior_data_tr)
        fm.write_sequence(high_port_filename_tr, high_port_data_tr)
        write_session_transitions(sessions_filename_tr, session_transitions_tr)
        print(f"Generated training data for run_{next_run} with {args.train_pattern}-step pattern")
    
    # Validation data with 13-step pattern
    behavior_filename_v = fm.get_experiment_file("behavior_run_{}.txt", next_run, "v", subdir='seqs')
    high_port_filename_v = fm.get_experiment_file("high_port_run_{}.txt", next_run, "v", subdir='seqs')
    sessions_filename_v = fm.get_experiment_file("session_transitions_run_{}.txt", next_run, "v", subdir='seqs')
    
    if fm.check_files_exist(behavior_filename_v, high_port_filename_v, sessions_filename_v, verbose=False) and (not args.no_overwrite):
        print(f"Validation files already exist for run_{next_run}, skipping data generation")
    else:
        behavior_data_v, high_port_data_v, session_transitions_v = generate_patterned_data(
            args.num_steps_val, task_params, pattern_length=args.val_pattern)
        
        fm.write_sequence(behavior_filename_v, behavior_data_v)
        fm.write_sequence(high_port_filename_v, high_port_data_v)
        write_session_transitions(sessions_filename_v, session_transitions_v)
        print(f"Generated validation data for run_{next_run} with {args.val_pattern}-step pattern")
    
    # Write metadata
    metadata_filename = fm.get_experiment_file("metadata.txt", next_run)
    with open(metadata_filename, 'a') as meta_file:
        meta_file.write(f"Run {next_run}\n")
        meta_file.write(f"Training data: {args.num_steps_train:,} steps with {args.train_pattern}-step pattern\n")
        meta_file.write(f"Validation data: {args.num_steps_val:,} steps with {args.val_pattern}-step pattern\n")
        meta_file.write(f"Task parameters:\n")
        for domain, params in task_params.items():
            meta_file.write(f"{' '*2}{domain}\n")
            meta_file.write(f"{' '*2}Environment parameters:\n{' '*4}{params['environment']}\n")
            meta_file.write(f"{' '*2}Agent parameters:\n{' '*4}{params['agent']}\n")
        meta_file.write(f"\n")
    
    print(f"Metadata saved to {metadata_filename}")

if __name__ == "__main__":
    main() 