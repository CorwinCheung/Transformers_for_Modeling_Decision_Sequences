import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import numpy as np
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts
import os
import glob

def generate_data(num_steps, agent, environment):
    behavior_data = []
    high_port_data = []
    # Initialize high port based on the environment's first_bit
    high_port = 1 if environment.first_bit else 0

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

def write_file(filename, data):
    with open(filename, 'w') as f:
        for i, token in enumerate(data):
            if i % 100 == 0:
                f.write('\n')
            f.write(token)

def main(num_steps=1000000, profile=False, include_val=True):
    
    environment = Original_2ABT_Spouts(0.8, 0.2, 0.02)
    agent = RFLR_mouse(alpha=-0.5, beta=2.1, tau=1.4, policy="greedy_policy")

    datasets = ['tr', 'v'] if include_val else ['tr']

    # Keep same root filenames to lock train and val sets to same ID.
    base_path = os.path.dirname(os.path.dirname(__file__))
    behavior_filename_base = find_filename(os.path.join(base_path, "data/2ABT_behavior.txt"))
    high_port_filename_base = find_filename(os.path.join(base_path, "data/2ABT_high_port.txt"))
    
    for suffix in datasets:
        behavior_filename = behavior_filename_base.replace('.txt', f'{suffix}.txt')
        high_port_filename = high_port_filename_base.replace('.txt', f'{suffix}.txt')
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

        with open(os.path.join(base_path, "data/metadata.txt"), 'a') as meta_file:
            meta_file.write(f"\nData files: {behavior_filename}, {high_port_filename}\n")
            meta_file.write(f"Number of steps: {num_steps:,}\n")
            meta_file.write(f"Environment parameters: high_reward_prob={environment.high_reward_prob}, "
                            f"low_reward_prob={environment.low_reward_prob}, "
                            f"transition_prob={environment.transition_prob}\n")
            meta_file.write(f"Agent parameters: alpha={agent.alpha}, beta={agent.beta}, tau={agent.tau}\n")
            meta_file.write(f"Agent policy: {agent.policy}\n")

        print(f"Generated {num_steps} steps of behavior data and saved to {behavior_filename}")
        print(f"Generated {num_steps} steps of high port data and saved to {high_port_filename}")
    print(f"Metadata saved to {os.path.join(base_path, 'data/metadata.txt')}")

if __name__ == "__main__":
    # Set profile to True to enable profiling, False to skip
    main(profile=False)
