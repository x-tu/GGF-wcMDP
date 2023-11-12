"""Common functions used."""

import numpy as np


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MDP4LP:
    """General-purpose MDP data (S, A, D, T, C/R, gamma, W) for solving LP problems.

    Explicitly, the MDP data includes:
        - S: number of states
        - A: number of actions
        - D: number of groups
        - T: transition matrix
        - C/R: cost/reward matrix
        - gamma: discount factor
        - W: weight vector
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_groups: int,
        transition: np.array,
        costs: np.array,
        discount: float,
        weights: np.array,
        rewards: np.array = None,
        minimize: bool = True,
    ):
        self.state_indices = np.arange(num_states)
        self.action_indices = np.arange(num_actions)
        self.group_indices = np.arange(num_groups)
        self.transition = transition
        self.costs = costs
        self.rewards = rewards
        self.discount = discount
        # add support for rewards
        if rewards:
            # double-check the minimize flag
            assert minimize == False, "Rewards are only used in maximize problems."
            # maximizing rewards is equivalent to minimizing negative rewards
            self.costs = -rewards
        self.weights = weights
