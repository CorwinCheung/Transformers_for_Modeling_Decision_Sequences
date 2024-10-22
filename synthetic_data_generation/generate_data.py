import numpy as np
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts
import os

def generate_data(num_steps, agent, environment):
    """
    Generate data of agent's behavior in the environment.

    Args:
    - num_steps (int): Number of steps to simulate.
    - agent (MouseAgent): The agent that makes decisions.
    - environment (WaterSpoutEnvironment): The environment in which the agent operates.

    Returns:
    - data (list): A list of strings representing the agent's behavior ('L', 'R', 'l', 'r').
    """
    data = []

    if environment.first_bit:
        data.append('O') #starts on right

    for step in range(num_steps):
        choice = agent.make_choice()

        reward, swapped = environment.step(choice)

        if swapped: #swap occurs, log it as S
            data.append('S')
        if choice == 0:  # Left choice
            if reward:
                data.append('L')
            else:
                data.append('l')
        else:  # Right choice
            if reward:
                data.append('R')
            else:
                data.append('r')

        agent.update_phi(choice,reward)

    return data

def main():
    num_steps = 10000
    # num_steps = 20 

    environment = Original_2ABT_Spouts(0.8,0.2,0.02)
    agent = RFLR_mouse(alpha=0.75, beta=2.1, tau=1.4)

    behavior_data = generate_data(num_steps, agent, environment)

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
    filename = find_filename("../data/2ABT_logistic.txt")

    with open(filename, 'w') as f:
        counter = 0
        for token in behavior_data:
            if counter == 100:
                f.write('\n')
                counter = 0
            f.write(token)
            counter += 1
            


    print(f"Generated {num_steps} steps of behavior data and saved to {filename}")
    with open("../data/metadata.txt", 'a') as meta_file:
        meta_file.write(f"\nData file: {filename}\n")
        meta_file.write(f"Number of steps: {num_steps:,}\n") 
        meta_file.write(f"Environment parameters: high_reward_prob={environment.high_reward_prob}, low_reward_prob={environment.low_reward_prob}, swap_prob={environment.swap_prob}\n")
        meta_file.write(f"Agent parameters: alpha={agent.alpha}, beta={agent.beta}, tau={agent.tau}\n")
    
    print(f"Metadata saved to ../data/metadata.txt")
main()