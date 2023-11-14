"""Common functions used."""

import numpy as np

from utils.encoding import state_int_index_to_vector


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
        encoding_int: bool = True,
        base_num_states: int = None,
    ):
        if not encoding_int:
            assert base_num_states is not None, "base_num_states must be provided."
            self.state_tuple_list = [
                tuple(
                    state_int_index_to_vector(i, num_groups, base_num_states).tolist()
                )
                for i in range(num_states)
            ]

        self.state_indices = np.arange(num_states)
        self.action_indices = np.arange(num_actions)
        self.group_indices = np.arange(num_groups)
        self.transition = transition
        self.costs = costs
        self.rewards = rewards
        self.discount = discount
        self.encoding_int = encoding_int
        # add support for rewards
        if rewards:
            # double-check the minimize flag
            assert minimize == False, "Rewards are only used in maximize problems."
            # maximizing rewards is equivalent to minimizing negative rewards
            self.costs = -rewards
        self.weights = weights

        # validate the MDP data
        self.validate_transition()
        self.validate_costs()

    def validate_transition(self):
        """Validate the transition matrix."""
        assert self.transition.shape == (
            len(self.state_indices),
            len(self.state_indices),
            len(self.action_indices),
        ), "The transition matrix must be a square matrix."
        assert np.all(self.transition >= 0) & np.all(
            self.transition <= 1
        ), "The transition matrix must be a stochastic matrix."
        assert np.all(
            abs(np.sum(self.transition, axis=1) - 1.0) < 1e-4
        ), "Each row of the transition matrix must sum to 1."

    def validate_costs(self):
        """Validate the cost matrix."""
        assert self.costs.shape == (
            len(self.state_indices),
            len(self.action_indices),
            len(self.group_indices),
        ), "The cost matrix must be a 3D matrix."
        assert np.all(self.costs >= 0), "The cost matrix must be non-negative."
        assert np.all(
            np.sum(self.costs, axis=2) >= 0
        ), "The cost matrix must be non-negative."
