import numpy as np


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
