import numpy as np
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts
import os

def generate_data(num_steps, agent, environment):
    """
    Generate data of the agent's behavior in the environment and the high port at each step.

    Args:
    - num_steps (int): Number of steps to simulate.
    - agent (MouseAgent): The agent that makes decisions.
    - environment (WaterSpoutEnvironment): The environment in which the agent operates.

    Returns:
    - behavior_data (list): A list of strings representing the agent's behavior ('L', 'R', 'l', 'r').
    - high_port_data (list): A list of '0's and '1's indicating the high port at each time step (0: left, 1: right).
    """
    behavior_data = []
    high_port_data = []
    # Initialize high port based on the environment's first_bit
    high_port = 1 if environment.first_bit else 0

    for step in range(num_steps):
        choice = agent.make_choice()
        reward, transitioned = environment.step(choice)
        high_port_data.append(str(high_port))
        if choice == 0:  # Left choice
            behavior_data.append('L' if reward else 'l')
        else:  # Right choice
            behavior_data.append('R' if reward else 'r')
        # Update high port if a transition occurred
        if transitioned:
            high_port = 1 - high_port  # Switch high port
        agent.update_phi(choice, reward)
    return behavior_data, high_port_data

def main():
    num_steps = 10000

    environment = Original_2ABT_Spouts(0.8, 0.2, 0.02)
    agent = RFLR_mouse(alpha=0.75, beta=2.1, tau=1.4, policy="probability_matching")

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

    # Write behavior data to file
    with open(behavior_filename, 'w') as f:
        counter = 0
        for token in behavior_data:
            if counter == 100:
                f.write('\n')
                counter = 0
            f.write(token)
            counter += 1

    # Write high port data to file
    with open(high_port_filename, 'w') as f:
        counter = 0
        for token in high_port_data:
            if counter == 100:
                f.write('\n')
                counter = 0
            f.write(token)
            counter += 1

    print(f"Generated {num_steps} steps of behavior data and saved to {behavior_filename}")
    print(f"Generated {num_steps} steps of high port data and saved to {high_port_filename}")

    # Save metadata
    with open("../data/metadata.txt", 'a') as meta_file:
        meta_file.write(f"\nData files: {behavior_filename}, {high_port_filename}\n")
        meta_file.write(f"Number of steps: {num_steps:,}\n")
        meta_file.write(f"Environment parameters: high_reward_prob={environment.high_reward_prob}, "
                        f"low_reward_prob={environment.low_reward_prob}, "
                        f"transition_prob={environment.transition_prob}\n")
        meta_file.write(f"Agent parameters: alpha={agent.alpha}, beta={agent.beta}, tau={agent.tau}\n")
        meta_file.write(f"Agent policy: {agent.policy}\n")

    print(f"Metadata saved to ../data/metadata.txt")

if __name__ == "__main__":
    main()
