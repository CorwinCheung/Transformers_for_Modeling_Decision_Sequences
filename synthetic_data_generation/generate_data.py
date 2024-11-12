import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import numpy as np
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts
import os

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

def main():
    num_steps = 1000000

    environment = Original_2ABT_Spouts(0.8, 0.2, 0.02)
    agent = RFLR_mouse(alpha=-0.5, beta=2.1, tau=1.4, policy="greedy_policy")

    behavior_data, high_port_data = generate_data(num_steps, agent, environment)

    def find_filename(base):
        if not os.path.exists(base):
            return base
        file_root, file_ext = os.path.splitext(base)
        counter = 2
        new_filename = f"{file_root}_run_{counter}{file_ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{file_root}_run_{counter}{file_ext}"
        return new_filename

    behavior_filename = find_filename("../data/2ABT_behavior.txt")
    high_port_filename = find_filename("../data/2ABT_high_port.txt")

    # # Write behavior data to file
    # with open(behavior_filename, 'w') as f:
    #     counter = 0
    #     for token in behavior_data:
    #         if counter == 100:
    #             f.write('\n')
    #             counter = 0
    #         f.write(token)
    #         counter += 1

    # # Write high port data to file
    # with open(high_port_filename, 'w') as f:
    #     counter = 0
    #     for token in high_port_data:
    #         if counter == 100:
    #             f.write('\n')
    #             counter = 0
    #         f.write(token)
    #         counter += 1

    print(f"Generated {num_steps} steps of behavior data and saved to {behavior_filename}")
    print(f"Generated {num_steps} steps of high port data and saved to {high_port_filename}")

    # Save metadata
    # with open("../data/metadata.txt", 'a') as meta_file:
    #     meta_file.write(f"\nData files: {behavior_filename}, {high_port_filename}\n")
    #     meta_file.write(f"Number of steps: {num_steps:,}\n")
    #     meta_file.write(f"Environment parameters: high_reward_prob={environment.high_reward_prob}, "
    #                     f"low_reward_prob={environment.low_reward_prob}, "
    #                     f"transition_prob={environment.transition_prob}\n")
    #     meta_file.write(f"Agent parameters: alpha={agent.alpha}, beta={agent.beta}, tau={agent.tau}\n")
    #     meta_file.write(f"Agent policy: {agent.policy}\n")

    # print(f"Metadata saved to ../data/metadata.txt")

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime')  # Sort by cumulative time
    function_names = []
    cumulative_times = []

    # Extract top functions based on cumulative time
    for func, stat in stats.stats.items():
        filename, lineno, func_name = func
        cumulative_time = stat[3]  # cumulative time is the 4th element in stat tuple
        # Filter out irrelevant low-level functions by setting a threshold
        if cumulative_time > 0.01:  # Adjust threshold as needed
            function_names.append(f"{lineno}({func_name})")
            cumulative_times.append(cumulative_time)

    # Plot the profiling results
    plt.figure(figsize=(10, 6))
    plt.barh(function_names, cumulative_times, color="skyblue")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Function")
    plt.title("Cumulative Time of Key Functions in Profiled Code")
    plt.gca().invert_yaxis()
    plt.show()
   