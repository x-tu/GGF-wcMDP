"""This module contains class used to generate parameters for a single machine in MRP."""

import numpy as np


class CostReward:
    """Define the cost function for a single machine."""

    def __init__(self, num_states: int, rccc_wrt_max=1.5):
        """Initialize the cost function.

        Assumption: there are only two types of actions: 0 (do nothing) and 1 (replace).

        Args:
            num_states (`int`): number of states.
            rccc_wrt_max (`float`): ratio of the Replacement Cost Constant Coefficient
                w.r.t the max cost in passive mode.
        """

        self.num_s = num_states
        self.costs = self.get_costs(rccc_wrt_max=rccc_wrt_max)
        self.rewards = 1 - self.costs

    def get_costs(self, rccc_wrt_max: float = 1.5) -> np.array:
        """Define the cost function in size [S, A]."""

        costs = np.zeros([self.num_s, 2])
        # define the cost of doing nothing
        costs[:, 0] = self.get_cost_by_type(cost_type="quadratic")
        # define the cost of replacement by the ratio
        costs[:, 1] = self.get_cost_by_type(cost_type="rccc", rccc_wrt_max=rccc_wrt_max)
        return costs / np.max(costs)

    def get_cost_by_type(self, cost_type: str, rccc_wrt_max: float = 1.5) -> np.array:
        """Define the cost function in size [S, D] under given action.

        Args:
            cost_type (`str`): the type of cost function.
            rccc_wrt_max (`float`): Replacement Cost Constant Coefficient.

        Returns:
            action_costs (`np.array`): conditional cost given action for each state and each group.
        """

        # type validation
        assert cost_type in ["constant", "linear", "quadratic", "rccc", "random"]
        # define a dictionary to map cost types to their corresponding calculations
        cost_type_mapping = {
            "constant": np.full(shape=self.num_s, fill_value=1),
            "linear": np.linspace(start=0, stop=self.num_s - 1, num=self.num_s),
            "quadratic": np.linspace(start=0, stop=self.num_s - 1, num=self.num_s) ** 2,
            "rccc": np.full(
                shape=self.num_s, fill_value=rccc_wrt_max * (self.num_s - 1) ** 2
            ),
            "random": np.random.rand(self.num_s),
        }
        return cost_type_mapping.get(cost_type, np.zeros(self.num_s))


class TransitionMatrix:
    """Define the Markov chain transition matrix for multiple groups."""

    def __init__(
        self, num_states: int, prob_remain: float = 0.5, deterioration_step: int = 1
    ):
        """Initialize the transition matrix in size [S, S, A].

        Assumption: there are only two types of actions: 0 (do nothing) and 1 (replace).

        Args:
            num_states (`int`): number of states
            prob_remain (`float`): probability of remaining in the same state for each machine in each state
            deterioration_step (`int`): number of steps to deteriorate
        """

        self.num_s = num_states
        self.transitions = np.zeros([self.num_s, self.num_s, 2])
        # define the transition matrix for doing nothing (deterioration)
        self.transitions[:, :, 0] = self.deterioration(
            prob_remain=prob_remain, deterioration_step=deterioration_step
        )
        # define the transition matrix for replacement
        self.transitions[:, :, 1] = self.pure_reset(prob_remain=prob_remain)

    def pure_reset(self, prob_remain: float) -> np.array:
        """Define the transition matrix for replacement.

        Args:
            prob_remain (`float`): probability of remaining at state 0 after replacement

        Returns:
            replace_transitions (`np.array`): transition matrix for replacement
        """

        replacement_transition = np.zeros([self.num_s, self.num_s])
        # staying at new state
        replacement_transition[:, 0] = prob_remain
        # going to deterioration
        replacement_transition[:, 1] = 1 - prob_remain
        return replacement_transition

    def deterioration(self, prob_remain: float, deterioration_step) -> np.array:
        """Define the transition matrix for deterioration.

        Args:
            prob_remain (`float`): probability of remaining at the same state after deterioration
            deterioration_step (`int`): number of steps to deteriorate
        """

        deterioration_transitions = np.eye(self.num_s) * prob_remain
        # last step
        deterioration_transitions[-1, -1] = 1
        deterioration_prob_step = (1 - prob_remain) / deterioration_step
        for s in range(self.num_s - 1):
            remaining_steps = min(deterioration_step, self.num_s - s - 1)
            deterioration_transitions[
                s, s + 1 : s + 1 + remaining_steps
            ] = deterioration_prob_step
            deterioration_transitions[s, -1] += (
                deterioration_step - remaining_steps
            ) * deterioration_prob_step
        # data validation
        assert deterioration_transitions.sum(axis=1).all() == 1
        return deterioration_transitions
