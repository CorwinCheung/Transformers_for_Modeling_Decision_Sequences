import random

class Original_2ABT_Spouts:
    def __init__(self, high_reward_prob=0.8, low_reward_prob=0.2, swap_prob=0.02):
        """
        Initializes the environment.
        
        Args:
        - high_reward_prob (float): Probability of receiving a reward from the high-reward spout.
        - low_reward_prob (float): Probability of receiving a reward from the low-reward spout.
        - swap_prob (float): Probability that the spouts will swap locations at each step.
        """
        self.high_reward_prob = high_reward_prob
        self.low_reward_prob = low_reward_prob
        self.swap_prob = swap_prob
        
        # Initial positions of the spouts (0 = left, 1 = right)
        self.high_spout_position = random.random() < 0.5
        self.low_spout_position = 1  - self.high_spout_position 
        self.first_bit = self.high_spout_position

    def _swap_spouts(self):
        """Randomly swap the spouts with a given probability."""
        if random.random() < self.swap_prob:
            self.high_spout_position, self.low_spout_position = self.low_spout_position, self.high_spout_position
            return True
        return False

    def step(self, choice):
        """
        Simulate one step in the environment.
        
        Args:
        - choice (int): The agent's choice of spout (0 = left, 1 = right).
        
        Returns:
        - reward (bool): Whether the agent received a reward (True or False).
        """
        swapped = self._swap_spouts()

        selected_high_reward = (choice == self.high_spout_position)

        if selected_high_reward:
            reward = random.random() < self.high_reward_prob
        else:
            reward = random.random() < self.low_reward_prob

        return reward, swapped

    def get_spout_positions(self):
        """Returns the current positions of the high and low reward spouts."""
        return self.high_spout_position, self.low_spout_position
