import numpy as np

class RFLR_mouse:
    def __init__(self, alpha=0.5, beta=2, tau=1.2):
        """
        Initialize the agent with the provided parameters.

        Args:
        - alpha (float): Weight for the most recent choice.
        - beta (float): Weight for the reward-based information.
        - tau (float): Time constant for decaying the influence of past choices/rewards.
        """
        self.alpha = alpha  # influence of the most recent choice
        self.beta = beta    # influence of the reward
        self.tau = tau      # decay rate for the reward history
        self.phi_t = 0.5      # initial value of the recursive reward term
        self.last_choice = np.random.choice([0,1])

    def update_phi(self, c_t, r_t):
        """
        Update the recursive reward term (phi_t) based on the current choice and reward.

        Args:
        - c_t (int): Current choice (0 for left, 1 for right).
        - r_t (int): Current reward (1 for rewarded, 0 for no reward).
        """
        self.phi_t = self.beta * (2 * c_t -1) * r_t + np.exp(-1 / self.tau) * self.phi_t

    def compute_log_odds(self, c_t):
        """
        Compute the log-odds for the next choice based on the recursive formula.

        Args:
        - c_t (int): Current choice (0 for left, 1 for right).

        Returns:
        - log_odds (float): Log-odds of selecting the action on the next trial.
        """
        log_odds = self.alpha * (2 * c_t -1) + self.phi_t
        return log_odds

    def make_choice(self):
        """
        Make a choice based on the current log-odds of selecting the left or right spout.

        The agent uses a sigmoid function to convert the log-odds into probabilities.

        Returns:
        - choice (int): The choice made by the agent (0 for left, 1 for right).
        """

        log_odds = self.compute_log_odds(self.last_choice)

        prob_right = 1 / (1 + np.exp(-log_odds))

        choice = np.random.choice([0, 1], p=[1 - prob_right, prob_right])
        #deterministic environment - reward 100% on the right <->
        #greedy policy. Expect the adjusted accuracy to be even higher. Perfectly
        #predictable, learn the algorithm. tasking the transformer

        self.last_choice = choice

        return choice

def main():
    agent = RFLR_mouse(alpha=0.5, beta=2, tau=1.2)

main()