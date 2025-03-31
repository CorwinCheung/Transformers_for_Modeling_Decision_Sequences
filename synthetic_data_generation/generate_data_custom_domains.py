import configparser
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.file_management as fm
from pprint import pformat

# Import functions from generate_data to avoid duplication
from synthetic_data_generation.generate_data import (
    generate_data, initialize_logger, write_session_transitions, load_param_sets
)

def parse_args():
    """Parse command-line arguments for custom domain data generation"""
    import argparse
    parser = argparse.ArgumentParser(description="Generate data with custom domain selections")
    parser.add_argument('--run', type=int, default=None,
                      help='Run number')
    parser.add_argument('--no_overwrite', action='store_false', default=True,
                      help='Prevent overwriting existing data')
    parser.add_argument('--num_steps_train', type=int, default=100000,
                      help='Training steps')
    parser.add_argument('--num_steps_val', type=int, default=1000000,
                      help='Validation steps')
    parser.add_argument('--train_domains', type=str, nargs='+', default=['B'],
                      help='Domains for training data')
    parser.add_argument('--val_domains', type=str, nargs='+', default=['A', 'C'],
                      help='Domains for validation data')
    parser.add_argument('--config_file', type=str, default='three_domains.ini',
                      help='Config file for domains')
    return parser.parse_args()

def generate_data_custom_domains(num_steps, task_params, domains):
    """
    Generate data using only specified domains
    
    Raises:
        ValueError: If none of the specified domains are found
    """
    # Filter task_params to only include the specified domains
    filtered_params = {domain: task_params[domain] for domain in domains if domain in task_params}
    if not filtered_params:
        raise ValueError(f"None of the specified domains {domains} were found in the configuration")
    
    behavior_data = []
    high_port_data = []
    session_transitions = []
    trials_per_session = 1000  # Typical session length
    session_count = (num_steps + trials_per_session - 1) // trials_per_session
    
    for session in range(session_count):
        # Randomly select one of the filtered domains
        domain_id = random.choice(list(filtered_params.keys()))
        env_params = filtered_params[domain_id]['environment']
        agent_params = filtered_params[domain_id]['agent']
        
        # Setup environment and agent
        environment = Original_2ABT_Spouts(
            high_reward_prob=env_params['high_reward_prob'],
            low_reward_prob=env_params['low_reward_prob'],
            transition_prob=env_params['transition_prob']
        )
        agent = RFLR_mouse(
            alpha=agent_params['alpha'],
            beta=agent_params['beta'],
            tau=agent_params['tau'],
            policy=agent_params['policy']
        )
        
        # Calculate number of trials for this session
        session_trials = min(trials_per_session, num_steps - len(behavior_data))
        if session_trials <= 0:
            break
            
        # Generate session data
        session_behavior, session_high_port = generate_session(session_trials, agent, environment)
        
        # Record session transition and add data
        trial_num = len(behavior_data)
        session_transitions.append((trial_num, domain_id))
        behavior_data.extend(session_behavior)
        high_port_data.extend(session_high_port)
    
    return behavior_data, high_port_data, session_transitions

def main():
    """Generate synthetic data with separate domains for training/validation"""
    args = parse_args()
    
    # Get next run number and initialize logger
    next_run = args.run or (fm.get_latest_run() + 1)
    initialize_logger(next_run)
    
    # Load domain configurations
    task_params = load_param_sets(args.config_file)
    run_dir = fm.ensure_run_dir(next_run, subdir='seqs')
    
    # Training data with specified domains
    behavior_filename_tr = fm.get_experiment_file("behavior_run_{}.txt", next_run, "tr", subdir='seqs')
    high_port_filename_tr = fm.get_experiment_file("high_port_run_{}.txt", next_run, "tr", subdir='seqs')
    sessions_filename_tr = fm.get_experiment_file("session_transitions_run_{}.txt", next_run, "tr", subdir='seqs')
    
    if fm.check_files_exist(behavior_filename_tr, high_port_filename_tr, sessions_filename_tr, verbose=False) and (not args.no_overwrite):
        print(f"Training files already exist for run_{next_run}, skipping data generation")
    else:
        behavior_data_tr, high_port_data_tr, session_transitions_tr = generate_data_custom_domains(
            args.num_steps_train, task_params, args.train_domains)
        
        fm.write_sequence(behavior_filename_tr, behavior_data_tr)
        fm.write_sequence(high_port_filename_tr, high_port_data_tr)
        write_session_transitions(sessions_filename_tr, session_transitions_tr)
        print(f"Generated training data for run_{next_run} using domains {args.train_domains}")
    
    # Validation data with different domains
    behavior_filename_v = fm.get_experiment_file("behavior_run_{}.txt", next_run, "v", subdir='seqs')
    high_port_filename_v = fm.get_experiment_file("high_port_run_{}.txt", next_run, "v", subdir='seqs')
    sessions_filename_v = fm.get_experiment_file("session_transitions_run_{}.txt", next_run, "v", subdir='seqs')
    
    if fm.check_files_exist(behavior_filename_v, high_port_filename_v, sessions_filename_v, verbose=False) and (not args.no_overwrite):
        print(f"Validation files already exist for run_{next_run}, skipping data generation")
    else:
        behavior_data_v, high_port_data_v, session_transitions_v = generate_data_custom_domains(
            args.num_steps_val, task_params, args.val_domains)
        
        fm.write_sequence(behavior_filename_v, behavior_data_v)
        fm.write_sequence(high_port_filename_v, high_port_data_v)
        write_session_transitions(sessions_filename_v, session_transitions_v)
        print(f"Generated validation data for run_{next_run} using domains {args.val_domains}")
    
    # Write metadata
    metadata_filename = fm.get_experiment_file("metadata.txt", next_run)
    with open(metadata_filename, 'a') as meta_file:
        meta_file.write(f"Run {next_run}\n")
        meta_file.write(f"Training data: {args.num_steps_train:,} steps using domains {args.train_domains}\n")
        meta_file.write(f"Validation data: {args.num_steps_val:,} steps using domains {args.val_domains}\n")
        meta_file.write(f"Task parameters:\n")
        for domain, params in task_params.items():
            if domain in args.train_domains or domain in args.val_domains:
                meta_file.write(f"{' '*2}{domain}\n")
                meta_file.write(f"{' '*2}Environment parameters:\n{' '*4}{params['environment']}\n")
                meta_file.write(f"{' '*2}Agent parameters:\n{' '*4}{params['agent']}\n")
        meta_file.write(f"\n")
    
    print(f"Metadata saved to {metadata_filename}")

if __name__ == "__main__":
    # Import here to avoid circular imports
    from synthetic_data_generation.agent import RFLR_mouse
    from synthetic_data_generation.environment import Original_2ABT_Spouts
    from synthetic_data_generation.generate_data import generate_session
    main()
