""""""
import numpy as np


class FairWeight:
    """Define the fairness weight for each arm."""

    def __init__(
        self, num_groups: int, weight_coefficient: int, weights: np.array = None
    ):
        # set the default weights if provided
        if weights and len(weights) == num_groups:
            self.weights = weights / np.sum(weights)
        # if the weights are not provided or the size is unmatched, generate them
        elif np.isscalar(weight_coefficient):
            self.weights = np.array(
                [1 / (weight_coefficient ** i) for i in range(num_groups)]
            )
            self.weights = self.weights / np.sum(self.weights)
        else:
            raise TypeError(
                "Please provide a scalar `weight_coefficient` "
                "or an array `weights` matching the size of the reward space."
            )


def calculate_ggi_reward(weights, n_rewards) -> float:
    """Calculate the GGI reward.

    Args:
        weights (list): weights of the arms
        n_rewards (list): rewards of the arms

    Returns:
        ggi_reward (float): the GGI reward
    """

    # assign the largest value to the smallest weight
    weights = sorted(weights, reverse=True)
    n_rewards = sorted(n_rewards, reverse=False)
    ggi_reward = np.dot(weights, n_rewards)
    return ggi_reward
