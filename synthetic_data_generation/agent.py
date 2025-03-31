import numpy as np


class RFLR_mouse:
    """Reinforcement Learning Forager agent for two-armed bandit task"""
    def __init__(self, alpha=0.5, beta=2, tau=1.2, policy="probability_matching"):
        """
        Initialize the agent with the provided parameters.

        Args:
            alpha (float): Weight for the most recent choice
            beta (float): Weight for the reward-based information
            tau (float): Time constant for decaying influence of past choices/rewards
            policy (str): Strategy for action selection ('probability_matching' or 'greedy_policy')
        """
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.phi_t = 0.5
        self.last_choice = np.random.choice([0, 1])
        self.policy = policy

    def update_phi(self, c_t, r_t):
        """
        Update the recursive reward term (phi_t) based on the current choice and reward.

        Args:
            c_t (int): Current choice (0 for left, 1 for right)
            r_t (int): Current reward (1 for rewarded, 0 for no reward)
        """
        self.phi_t = self.beta * (2 * c_t - 1) * r_t + np.exp(-1 / self.tau) * self.phi_t

    def compute_log_odds(self, c_t):
        """
        Compute the log-odds for the next choice based on the recursive formula.

        Args:
            c_t (int): Current choice (0 for left, 1 for right)

        Returns:
            float: Log-odds of selecting the right action on the next trial
        """
        log_odds = self.alpha * (2 * c_t - 1) + self.phi_t
        return log_odds

    def sigmoid(self, log_odds):
        """Convert log-odds to probability"""
        return 1 / (1 + np.exp(-log_odds))

    def make_choice(self):
        """Make choice based on policy and current value estimates"""
        log_odds = self.compute_log_odds(self.last_choice)
        prob_right = self.sigmoid(log_odds)

        if self.policy == "probability_matching":
            # Make a choice with probability matching the calculated value
            choice = np.random.choice([0, 1], p=[1 - prob_right, prob_right])
        elif self.policy == "greedy_policy":
            # Always select the higher-value option
            choice = 1 if prob_right > 0.5 else 0
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        self.last_choice = choice
        return choice

def main():
    agent = RFLR_mouse(alpha=0.75, beta=2.1, tau=1.4)

main()