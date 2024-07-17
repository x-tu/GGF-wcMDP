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

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_groups = num_groups
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


def get_identifier(params):
    g_string = (
        f"{params.machine_range[0]}-{params.machine_range[1]}"
        if params.machine_range
        else params.num_groups
    )
    k_string = (
        f"{params.resource_range[0]}-{params.resource_range[1]}"
        if params.resource_range
        else params.budget
    )
    return (
        f"G{g_string}_"
        f"C{params.cost_type_operation[:2]}-{params.cost_type_replace[:2]}_"
        f"F{'o' if params.ggi else 'x'}_"
        f"K{k_string}{'o' if params.force_to_use_all_resources else 'x'}"
    )


def get_default_weights(num_groups):
    weights = np.array([1 / (2**i) for i in range(num_groups)])
    return weights / np.sum(weights)


def update_params(params, num_groups, budget=None):
    # update identifier
    parts = params.identifier.split("_")
    parts[0] = f"G{num_groups}"
    if budget is not None:
        parts[
            -1
        ] = f"K{budget}{parts[-1][-1]}"  # Keep the last character ('o' or 'x') from the original identifier
    params.identifier = "_".join(parts)
    # update weights
    weights = np.array([1 / (2**i) for i in range(num_groups)])
    params.weights = weights / np.sum(weights)
    return params
