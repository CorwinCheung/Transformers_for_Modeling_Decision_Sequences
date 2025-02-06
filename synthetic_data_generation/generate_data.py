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
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import (ensure_run_dir, get_experiment_file,
                                   get_latest_run, write_sequence)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--num_steps', type=int, default=100000)

    # Agent parameters
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--policy', type=str, default=None)

    # Environment parameters
    parser.add_argument('--high_reward_prob', type=float, default=0.8)
    parser.add_argument('--low_reward_prob', type=float, default=0.2)
    parser.add_argument('--transition_prob', type=float, default=0.02)
    parser.add_argument('--multiple_contexts', type=bool, default=False,
                        help='Whether to vary parameters across sessions')
    parser.add_argument('--context_id', type=str, default=None,
                        help='shortcut to task parameters, overrides individual param args')
    return parser.parse_args()


def load_param_sets():
    config_path = os.path.join(os.path.dirname(__file__), 'sticky_unsticky_domains.ini')
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
    context_id = random.choice(list(task_params.keys()))
    env_params = task_params[context_id]['environment']
    agent_params = task_params[context_id]['agent']
    return context_id, env_params, agent_params


def configure_task_params(task_params, multiple_contexts=False):
    """Configure task parameters based on whether multiple contexts are used,
    or whether any parameters are provided.
    
    Args:
        task_params (dict): Dictionary containing task parameters for
        environment and agent. Each context should have environment and agent
        params. These can be None if no parameters are provided.
        multiple_contexts (bool): Whether to use multiple contexts/parameter
        sets.
        
    Returns:
        dict: Configured task parameters. If multiple_contexts is False,
        returns dict with single 'B' context. If True, returns original
        multi-context dict. Fills in defaults if not multiiple_contexts and no
        parameters are provided.
    """

    default_environment = {'high_reward_prob': 0.8,
                           'low_reward_prob': 0.2,
                           'transition_prob': 0.02}
    default_agent = {'alpha': 0.75,
                     'beta': 2.1,
                     'tau': 1.4,
                     'policy': 'probability_matching'}
    if (not multiple_contexts) and (not any([task_params.get(k, False) for k in ['A', 'B', 'C']])):
        task_params['environment'] = task_params.get('environment', default_environment)
        task_params['agent'] = task_params.get('agent', default_agent)
        task_params = {'B': task_params}
    return task_params


def generate_data(num_steps, task_params=None, multiple_contexts=False):
    """Generate data across multiple sessions with varying parameters."""
    behavior_data = []
    high_port_data = []
    context_transitions = []

    total_trials = 0
    trials_per_session_mean = 500
    trials_per_session_std = 100

    while total_trials < num_steps:
        # Generate number of trials for this session
        session_trials = int(np.random.normal(trials_per_session_mean, trials_per_session_std))
        session_trials = max(0, min(session_trials, num_steps - total_trials))  # Ensure reasonable bounds

        # Get parameters for this session
        context_id, env_params, agent_params = get_session_params(task_params)
        context_transitions.append((total_trials, context_id))
        # Initialize environment and agent for this session
        environment = Original_2ABT_Spouts(**env_params)
        agent = RFLR_mouse(**agent_params)

        # Generate session data
        session_behavior, session_high_port = generate_session(session_trials, agent, environment)

        # Append session data
        behavior_data.extend(session_behavior)
        high_port_data.extend(session_high_port)

        total_trials += session_trials

    return behavior_data, high_port_data, context_transitions


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


def write_context_transitions(filepath, context_transitions):
    """Write context transition data to file.
    
    Args:
        filepath (str): Path to output file
        context_transitions (list): List of (trial_number, context_id) tuples
    """
    with open(filepath, 'w') as f:
        for trial, context in context_transitions:
            f.write(f"{trial},{context}\n")


def main(
        run=None,
        num_steps=100000,
        profile=False,
        include_val=True,
        overwrite=False,
        task_params=None,
        multiple_contexts=False):

    # Get next run number.
    next_run = run or (get_latest_run() + 1)
    run_dir = ensure_run_dir(next_run, overwrite=overwrite, subdir='seqs')

    datasets = ['tr', 'v'] if include_val else ['tr']

    # Configure task parameters here to make logging easier.
    task_params = configure_task_params(task_params, multiple_contexts)

    for suffix in datasets:
        behavior_filename = get_experiment_file("behavior_run_{}.txt", next_run, suffix, subdir='seqs')
        high_port_filename = get_experiment_file("high_port_run_{}.txt", next_run, suffix, subdir='seqs')
        context_filename = get_experiment_file("context_transitions_run_{}.txt", next_run, suffix, subdir='seqs')
        if profile:
            with cProfile.Profile() as pr:
                behavior_data, high_port_data, context_transitions = generate_data(num_steps, task_params, multiple_contexts)
            stats = pstats.Stats(pr)
            stats.sort_stats('cumtime')
            plot_profiling_results(stats)
        else:
            behavior_data, high_port_data, context_transitions = generate_data(num_steps, task_params, multiple_contexts)

        write_sequence(behavior_filename, behavior_data)
        write_sequence(high_port_filename, high_port_data)
        write_context_transitions(context_filename, context_transitions)
        # Write metadata
        metadata_filename = get_experiment_file("metadata.txt", next_run)
        with open(metadata_filename, 'a') as meta_file:
            meta_file.write(f"Run {next_run}\n")
            meta_file.write(f"Dataset {suffix}\n")
            meta_file.write(f"Number of steps: {num_steps:,}\n")
            meta_file.write(f"Multiple contexts: {multiple_contexts:,}\n")
            meta_file.write(f"Task parameters:\n")
            for context, params in task_params.items():
                meta_file.write(f"{' '*2}{context}\n")
                meta_file.write(f"{' '*2}Environment parameters:\n{' '*4}{params['environment']}\n")
                meta_file.write(f"{' '*2}Agent parameters:\n{' '*4}{params['agent']}\n")
    
            # meta_file.write(f"Environment parameters: high_reward_prob={environment.high_reward_prob}, "
            #                 f"low_reward_prob={environment.low_reward_prob}, "
            #                 f"transition_prob={environment.transition_prob}\n")
            # meta_file.write(f"Agent parameters: alpha={agent.alpha}, beta={agent.beta}, tau={agent.tau}\n")
            # meta_file.write(f"Agent policy: {agent.policy}\n")
            meta_file.write(f"\n")

        print(f"Generated data for run_{next_run}")
        print(f"Files saved to {run_dir}")
    print(f"Metadata saved to {metadata_filename}")


if __name__ == "__main__":

    args = parse_args()
    print("Context ID:", args.context_id)
    if args.multiple_contexts or args.context_id:
        task_params = load_param_sets()
        if args.context_id:
            task_params = {args.context_id: task_params[args.context_id]}
            print(task_params)
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
         overwrite=args.overwrite,
         num_steps=args.num_steps,
         task_params=task_params,
         multiple_contexts=args.multiple_contexts)
