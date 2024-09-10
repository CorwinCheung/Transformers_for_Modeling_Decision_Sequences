import numpy as np

class RFLR_mouse:
    def __init__(self, alpha=0.5, beta=0.5, tau=1.0):
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
        self.phi_t = 0      # initial value of the recursive reward term
        self.choice_history = []  # store the history of choices (e.g., 0 for left, 1 for right)
        self.reward_history = []  # store the history of rewards (1 for reward, 0 for no reward)

    def update_phi(self, c_t, r_t):
        """
        Update the recursive reward term (phi_t) based on the current choice and reward.

        Args:
        - c_t (int): Current choice (0 for left, 1 for right).
        - r_t (int): Current reward (1 for rewarded, 0 for no reward).
        """
        self.phi_t = self.beta * c_t * r_t + np.exp(-1 / self.tau) * self.phi_t

    def compute_log_odds(self, c_t):
        """
        Compute the log-odds for the next choice based on the recursive formula.

        Args:
        - c_t (int): Current choice (0 for left, 1 for right).

        Returns:
        - log_odds (float): Log-odds of selecting the action on the next trial.
        """
        log_odds = self.alpha * c_t + self.phi_t
        return log_odds

    def make_choice(self):
        """
        Make a choice based on the current log-odds of selecting the left or right spout.

        The agent uses a sigmoid function to convert the log-odds into probabilities.

        Returns:
        - choice (int): The choice made by the agent (0 for left, 1 for right).
        """
        if not self.choice_history:
            # If it's the first trial, choose randomly
            return np.random.choice([0, 1])

        # Get the most recent choice
        last_choice = self.choice_history[-1]

        # Compute log-odds for the next choice
        log_odds = self.compute_log_odds(last_choice)

        # Convert log-odds to probability using the sigmoid function
        prob_right = 1 / (1 + np.exp(-log_odds))

        # Make a stochastic choice based on the computed probability
        choice = np.random.choice([0, 1], p=[1 - prob_right, prob_right])
        return choice

    def step(self, reward):
        """
        Perform one step of the agent's decision process.

        Args:
        - reward (int): Whether the agent received a reward (1 for rewarded, 0 for no reward).

        Returns:
        - choice (int): The choice made by the agent (0 for left, 1 for right).
        """
        # Make a new choice
        choice = self.make_choice()

        # Update the choice and reward history
        self.choice_history.append(choice)
        self.reward_history.append(reward)

        # Update the recursive reward term phi_t
        self.update_phi(choice, reward)

        return choice

def main():
    agent = RFLR_mouse(alpha=0.6, beta=0.9, tau=1.5)

main()