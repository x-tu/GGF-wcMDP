"""This module contains classes and functions for dealing with fairness."""

import numpy as np


class FairWeight:
    """Define the fairness weight for each group."""

    def __init__(self, num_groups: int, weight_type: str = "exponential2"):
        """Provide the weights for each group.

        Args:
            num_groups (int): number of groups.
            weight_type (str): type of the weights.
        """
        assert weight_type in [
            "uniform",
            "exponential2",
            "exponential3",
            "random",
        ], "The weight type is not supported."
        weight_mapping = {
            "uniform": np.array([1 / num_groups] * num_groups),
            "exponential2": np.array([1 / (2 ** i) for i in range(num_groups)]),
            "exponential3": np.array([1 / (3 ** i) for i in range(num_groups)]),
            "random": np.random.rand(num_groups),
        }
        self.weights = weight_mapping.get(weight_type)
        self.weights /= np.sum(self.weights)


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
