import numpy as np
from agent import RFLR_mouse
from environment import Original_2ABT_Spouts

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

    for step in range(num_steps):
        # The agent makes a choice (0 for left, 1 for right)
        choice = agent.make_choice()

        # The environment returns whether the choice results in a reward (1 for rewarded, 0 for no reward)
        reward = environment.step(choice)

        # Log behavior as 'L' or 'R' for rewarded choices, and 'l' or 'r' for unrewarded choices
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

        # Update the agent's state (pass the reward to the agent)
        agent.step(reward)

    return data

def main():
    # Set the number of steps/tokens to generate
    num_steps = 1000  # Adjust this value as needed

    # Initialize the environment and agent
    environment = Original_2ABT_Spouts()
    agent = RFLR_mouse(alpha=0.6, beta=0.9, tau=1.5)  # Customize alpha, beta, tau as needed

    # Generate data
    behavior_data = generate_data(num_steps, agent, environment)

    # Write data to a file
    with open('./data/mouse_behavior.txt', 'w') as f:
        for token in behavior_data:
            f.write(token + '\n')

    print(f"Generated {num_steps} steps of behavior data and saved to './data/mouse_behavior.txt'")
main()