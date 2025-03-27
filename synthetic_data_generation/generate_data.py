import configparser
import cProfile
import glob
import io
import os
import pstats
import random
# Add the project root directory to Python path
import sys

import matplotlib.pyplot as plt
import numpy as np
from .agent import RFLR_mouse
from .environment import Original_2ABT_Spouts

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the current directory to the path 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils.file_management as fm
from pprint import pformat

logger = None

def initialize_logger(run_number):
    global logger
    logger = fm.setup_logging(run_number, 'data_generation', 'generate_data')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--no_overwrite', action='store_false', default=True,
                        help='Pass flag to prevent overwriting existing data')
    parser.add_argument('--num_steps_train', type=int, default=100000)
    parser.add_argument('--num_steps_val', type=int, default=100000)
    # Agent parameters
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--policy', type=str, default=None)

    # Environment parameters
    parser.add_argument('--high_reward_prob', type=float, default=0.8)
    parser.add_argument('--low_reward_prob', type=float, default=0.2)
    parser.add_argument('--transition_prob', type=float, default=0.02)
    parser.add_argument('--multiple_domains', action='store_true', default=False,
                        help='Whether to vary parameters across sessions')
    parser.add_argument('--domain_id', type=str, default=None,
                        help='shortcut to task parameters, overrides individual param args')
    # Add a new argument for config file
    parser.add_argument('--config_file', type=str, default='domains.ini',
                        help='Configuration file for domains (domains.ini or three_domains.ini)')
    return parser.parse_args()


def load_param_sets(config_file='domains.ini'):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    config = configparser.ConfigParser()
    config.read(config_path)

    task_params = {}
    for section in config.sections():
        task_params[section] = {
            'environment': eval(config[section]['environment']) ,
            'agent': eval(config[section]['agent'])
        }
    return task_params


def generate_session(num_trials, agent, environment):
    """Generate data for a single session with specified number of trials."""
    behavior_data = []
    high_port_data = []
    high_port = environment.high_spout_position

    for step in range(num_trials):
        choice = agent.make_choice()
        reward, transitioned = environment.step(choice)
        if choice == 0:  # Left choice
            behavior_data.append('L' if reward else 'l')
        else:  # Right choice
            behavior_data.append('R' if reward else 'r')
        if transitioned:
            high_port = 1 - high_port
        high_port_data.append(str(high_port))
        agent.update_phi(choice, reward)

    return behavior_data, high_port_data


def get_session_params(task_params):
    """Randomly select parameters for a session from predefined sets."""

    # For now, keep environment and agent behaviors locked together, so choose
    # them together by ID
    domain_id = random.choice(list(task_params.keys()))
    env_params = task_params[domain_id]['environment']
    agent_params = task_params[domain_id]['agent']
    return domain_id, env_params, agent_params


def configure_task_params(task_params, multiple_domains=False, config_file='domains.ini'):
    """Configure task parameters based on whether multiple domains are used,
    or whether any parameters are provided.
    
    Args:
        task_params (dict): Dictionary containing task parameters for
        environment and agent. Each domain should have environment and agent
        params. These can be None if no parameters are provided.
        multiple_domains (bool): Whether to use multiple domains/parameter
        sets.
        config_file (str): Configuration file for domains (domains.ini or three_domains.ini)
        
    Returns:
        dict: Configured task parameters. If multiple_domains is False,
        returns dict with single 'B' domain. If True, returns original
        multi-domain dict. Fills in defaults if not multiple_cdomains and no
        parameters are provided.
    """

    # If task_params is empty, load from file
    if not task_params:
        task_params = load_param_sets(config_file)
        
    default_environment = {'high_reward_prob': 0.8,
                           'low_reward_prob': 0.2,
                           'transition_prob': 0.02}
    default_agent = {'alpha': 0.75,
                     'beta': 2.1,
                     'tau': 1.4,
                     'policy': 'probability_matching'}
    if (not multiple_domains) and (not any([task_params.get(k, False) for k in ['A', 'B', 'C']])):
        task_params['environment'] = task_params.get('environment', default_environment)
        task_params['agent'] = task_params.get('agent', default_agent)
        task_params = {'B': task_params}
    return task_params


def generate_data(num_steps, task_params=None, multiple_domains=False):
    """Generate data across multiple sessions with varying parameters."""
    behavior_data = []
    high_port_data = []
    session_transitions = []

    total_trials = 0
    trials_per_session_mean = 500
    trials_per_session_std = 100

    while total_trials < num_steps:
        # Generate number of trials for this session
        session_trials = int(np.random.normal(trials_per_session_mean, trials_per_session_std))
        session_trials = max(0, min(session_trials, num_steps - total_trials))  # Ensure reasonable bounds

        # Get parameters for this session
        domain_id, env_params, agent_params = get_session_params(task_params)
        session_transitions.append((total_trials, domain_id))
        # Initialize environment and agent for this session
        environment = Original_2ABT_Spouts(**env_params)
        agent = RFLR_mouse(**agent_params)

        # Generate session data
        session_behavior, session_high_port = generate_session(session_trials, agent, environment)

        # Append session data
        behavior_data.extend(session_behavior)
        high_port_data.extend(session_high_port)

        total_trials += session_trials

    return behavior_data, high_port_data, session_transitions


def plot_profiling_results(stats):
    function_names = []
    cumulative_times = []

    # Extract top functions based on cumulative time
    for func, stat in stats.stats.items():
        filename, lineno, func_name = func
        cumulative_time = stat[3]  # cumulative time is the 4th element in stat tuple
        if cumulative_time > 0.01:
            function_names.append(f"{lineno}({func_name})")
            cumulative_times.append(cumulative_time)

    # Plot the profiling results
    plt.figure(figsize=(10, 6))
    plt.barh(function_names, cumulative_times, color="skyblue")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Function")
    plt.title("Cumulative Time of Key Functions in Profiled Code")
    plt.gca().invert_yaxis()
    plt.savefig("cprofile of training")


def write_session_transitions(filepath, session_transitions):
    """Write session transition data (and if multiple_domains, domain transitions) to file.
    
    Args:
        filepath (str): Path to output file
        session_transitions (list): List of (trial_number, domain_id) tuples
    """
    with open(filepath, 'w') as f:
        for trial, domain in session_transitions:
            f.write(f"{trial},{domain}\n")


def main(
        run=None,
        num_steps_train=100000,
        num_steps_val=100000,
        profile=False,
        include_val=True,
        overwrite=True,
        task_params=None,
        multiple_domains=False,
        config_file='domains.ini'):

    initialize_logger(run)
    # Format the dictionary nicely
    args_str = pformat(locals(), indent=2)
    logger.info("Starting data generation with args:\n%s", args_str)

    # Get next run number.
    next_run = run or (fm.get_latest_run() + 1)
    run_dir = fm.ensure_run_dir(next_run, subdir='seqs')

    datasets = ['tr', 'v'] if include_val else ['tr']
    # If num_steps is a single value, repeat it for each dataset
    num_steps = [num_steps_train, num_steps_val] if include_val else [num_steps_train]

    # Configure task parameters here to make logging easier.
    task_params = configure_task_params(task_params, multiple_domains, config_file)

    for N, suffix in zip(num_steps, datasets):
        behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", next_run, suffix, subdir='seqs')
        high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", next_run, suffix, subdir='seqs')
        sessions_filename = fm.get_experiment_file("session_transitions_run_{}.txt", next_run, suffix, subdir='seqs')
        if fm.check_files_exist(behavior_filename, high_port_filename, sessions_filename, verbose=False) and (not overwrite):
            logger.info(f"Files already exist for run_{next_run}, skipping data generation")
            return None

        if profile:
            with cProfile.Profile() as pr:
                behavior_data, high_port_data, session_transitions = generate_data(N, task_params, multiple_domains)
            stats = pstats.Stats(pr)
            stats.sort_stats('cumtime')
            plot_profiling_results(stats)
        else:
            behavior_data, high_port_data, session_transitions = generate_data(N, task_params, multiple_domains)

        fm.write_sequence(behavior_filename, behavior_data)
        fm.write_sequence(high_port_filename, high_port_data)
        write_session_transitions(sessions_filename, session_transitions)
        # Write metadata
        metadata_filename = fm.get_experiment_file("metadata.txt", next_run)
        with open(metadata_filename, 'a') as meta_file:
            meta_file.write(f"Run {next_run}\n")
            meta_file.write(f"Dataset {suffix}\n")
            meta_file.write(f"Number of steps: {N:,}\n")
            meta_file.write(f"Multiple domains: {multiple_domains:,}\n")
            meta_file.write(f"Task parameters:\n")
            for domain, params in task_params.items():
                meta_file.write(f"{' '*2}{domain}\n")
                meta_file.write(f"{' '*2}Environment parameters:\n{' '*4}{params['environment']}\n")
                meta_file.write(f"{' '*2}Agent parameters:\n{' '*4}{params['agent']}\n")
            meta_file.write(f"\n")
        logger.info(f"Generated data for run_{next_run}")
        logger.info(f"Files saved to {run_dir}")
    logger.info(f"Metadata saved to {metadata_filename}")


if __name__ == "__main__":

    print('-' * 80)
    print('generate_data.py\n')

    args = parse_args()
    print("Domain ID:", args.domain_id)
    print("Config file:", args.config_file)
    
    if args.multiple_domains or args.domain_id:
        task_params = load_param_sets(args.config_file)
        if args.domain_id:
            task_params = {args.domain_id: task_params[args.domain_id]}
            print(pformat(task_params, indent=2))
    else:
        task_params = {}
        if all(v is None for v in [args.alpha, args.beta, args.tau, args.policy]):
            pass
        else:
            agent_params = {
                'alpha': args.alpha,
                'beta': args.beta,
                'tau': args.tau,
                'policy': args.policy
        }
            task_params['agent'] = agent_params

        if all(v is None for v in [args.high_reward_prob,
                                   args.low_reward_prob,
                                   args.transition_prob]):
            pass
        else:
            environment_params = {
                'high_reward_prob': args.high_reward_prob,
                'low_reward_prob': args.low_reward_prob,
                'transition_prob': args.transition_prob
            }
            task_params['environment'] = environment_params

    # Set profile to True to enable profiling, False to skip
    main(run=args.run,
         profile=args.profile,
         overwrite=args.no_overwrite,
         num_steps_train=args.num_steps_train,
         num_steps_val=args.num_steps_val,
         task_params=task_params,
         multiple_domains=args.multiple_domains,
         config_file=args.config_file)
