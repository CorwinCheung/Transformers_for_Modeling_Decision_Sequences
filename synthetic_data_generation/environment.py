import random

class Original_2ABT_Spouts:
    """Two-Armed Bandit environment with switching reward locations"""
    def __init__(self, high_reward_prob=0.8, low_reward_prob=0.2, transition_prob=0.02):
        """
        Initialize the environment.
        
        Args:
            high_reward_prob (float): Probability of receiving a reward from the high-reward spout
            low_reward_prob (float): Probability of receiving a reward from the low-reward spout
            transition_prob (float): Probability that the spouts will transition locations at each step
        """
        self.high_reward_prob = high_reward_prob
        self.low_reward_prob = low_reward_prob
        self.transition_prob = transition_prob
        
        # Initial positions (0=left, 1=right)
        self.high_spout_position = int(random.random() < 0.5)
        self.low_spout_position = 1 - self.high_spout_position
        self.first_bit = self.high_spout_position

    def _transition_spouts(self):
        """Randomly transition spouts based on transition probability"""
        if random.random() < self.transition_prob:
            self.high_spout_position, self.low_spout_position = self.low_spout_position, self.high_spout_position
            return True
        return False

    def step(self, choice):
        """
        Process agent's choice and deliver reward
        
        Args:
            choice (int): Agent's choice (0=left, 1=right)
            
        Returns:
            (reward, transitioned): Reward outcome and whether spouts switched
        """
        transitioned = self._transition_spouts()
        selected_high_reward = (choice == self.high_spout_position)

        if selected_high_reward:
            reward = random.random() < self.high_reward_prob
        else:
            reward = random.random() < self.low_reward_prob

        return reward, transitioned

    def get_spout_positions(self):
        """Return current positions of high and low reward spouts"""
        return self.high_spout_position, self.low_spout_position
