import cProfile
import glob
import io
import os
import pstats
# Add the project root directory to Python path
import sys

import matplotlib.pyplot as plt
import numpy as np
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import ensure_run_dir, get_latest_run


def generate_data(num_steps, agent, environment):
    behavior_data = []
    high_port_data = []
    # Initialize high port based on the environment's first_bit
    high_port = environment.high_spout_position

    for step in range(num_steps):
        choice = agent.make_choice()
        reward, transitioned = environment.step(choice)
        if choice == 0:  # Left choice
            behavior_data.append('L' if reward else 'l')
        else:  # Right choice
            behavior_data.append('R' if reward else 'r')
        # Update high port if a transition occurred
        if transitioned:
            high_port = 1 - high_port  # Switch high port
        high_port_data.append(str(high_port))
        agent.update_phi(choice, reward)
    return behavior_data, high_port_data


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


def find_filename(base, suffix='tr'):
    file_root, file_ext = os.path.splitext(base)
    counter = 0
    new_filename = f"{file_root}_run_{counter}*{file_ext}"
    while glob.glob(new_filename):
        counter += 1
        new_filename = f"{file_root}_run_{counter}*{file_ext}"
    return f"{file_root}_run_{counter}{file_ext}"


def write_file(filepath, data):
    with open(filepath, 'w') as f:
        for i, token in enumerate(data):
            if i % 100 == 0:
                f.write('\n')
            f.write(token)


def main(
        run=None,
        num_steps=100000,
        profile=False,
        include_val=True,
        overwrite=False,
        agent_params=None,
        environment_params=None):

    # Get next run number.
    next_run = run or (get_latest_run() + 1)
    run_dir = ensure_run_dir(next_run, overwrite=overwrite)

    # Set up environment and agent.
    if environment_params is None:
        environment_params = {'high_reward_prob': 0.8,
                              'low_reward_prob': 0.2,
                              'transition_prob': 0.02}
    environment = Original_2ABT_Spouts(**environment_params)

    if agent_params is None:
        agent_params = {'alpha': 0.75,
                        'beta': 2.1,
                        'tau': 1.4,
                        'policy': "probability_matching"}
    agent = RFLR_mouse(**agent_params)

    datasets = ['tr', 'v'] if include_val else ['tr']

    for suffix in datasets:
        behavior_filename = os.path.join(run_dir, f"behavior_run_{next_run}{suffix}.txt")
        high_port_filename = os.path.join(run_dir, f"high_port_run_{next_run}{suffix}.txt")

        if profile:
            with cProfile.Profile() as pr:
                behavior_data, high_port_data = generate_data(num_steps, agent, environment)
            stats = pstats.Stats(pr)
            stats.sort_stats('cumtime')
            plot_profiling_results(stats)
        else:
            behavior_data, high_port_data = generate_data(num_steps, agent, environment)

        write_file(behavior_filename, behavior_data)
        write_file(high_port_filename, high_port_data)

        # Write metadata
        metadata_path = os.path.join(run_dir, "metadata.txt")
        with open(metadata_path, 'a') as meta_file:
            meta_file.write(f"Run {next_run}\n")
            meta_file.write(f"Dataset {suffix}\n")
            meta_file.write(f"Number of steps: {num_steps:,}\n")
            meta_file.write(f"Environment parameters: high_reward_prob={environment.high_reward_prob}, "
                            f"low_reward_prob={environment.low_reward_prob}, "
                            f"transition_prob={environment.transition_prob}\n")
            meta_file.write(f"Agent parameters: alpha={agent.alpha}, beta={agent.beta}, tau={agent.tau}\n")
            meta_file.write(f"Agent policy: {agent.policy}\n")
            meta_file.write(f"\n")

        print(f"Generated data for run_{next_run}")
        print(f"Files saved to {run_dir}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
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
    args = parser.parse_args()

    if all(v is None for v in [args.alpha, args.beta, args.tau, args.policy]):
        agent_params = None
    else:
        agent_params = {
            'alpha': args.alpha,
            'beta': args.beta,
            'tau': args.tau,
            'policy': args.policy
        }

    if all(v is None for v in [args.high_reward_prob,
                               args.low_reward_prob,
                               args.transition_prob]):
        environment_params = None
    else:
        environment_params = {
            'high_reward_prob': args.high_reward_prob,
            'low_reward_prob': args.low_reward_prob,
            'transition_prob': args.transition_prob
        }   

    # Set profile to True to enable profiling, False to skip
    main(run=args.run, profile=args.profile, overwrite=args.overwrite,
         num_steps=args.num_steps, agent_params=agent_params,
         environment_params=environment_params)
