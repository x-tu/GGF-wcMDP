"""This module contains classes and functions for dealing with fairness."""

import numpy as np


class FairWeight:
    """Define the fairness weight for each group."""

    def __init__(
        self, num_groups: int, weight_coefficient: int = None, weights: np.array = None
    ):
        """Provide the weights for each group.

        Args:
            num_groups (int): number of groups.
            weight_coefficient (int): the coefficient of the weights.
            weights (np.array): the weights of the groups.
        """

        if weights:
            # check the size of the weights
            assert len(weights) == num_groups, "The size of the weights is unmatched."
            # normalize the weights if provided
            self.weights = weights / np.sum(weights)
        elif weight_coefficient:
            self.weights = np.array(
                [1 / (weight_coefficient ** i) for i in range(num_groups)]
            )
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = np.array([1 / num_groups] * num_groups)


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
