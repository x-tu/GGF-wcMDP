"""This module contains classes and functions for the MRP data generation."""

import itertools

import numpy as np

from utils.encoding import state_int_index_to_vector
from utils.ggf import FairWeight


class MRPData:
    def __init__(
        self,
        num_groups: int = 2,
        num_states: int = 3,
        num_actions: int = 2,
        rccc_wrt_max: float = 1.5,
        prob_remain: float = 0.8,
        deterioration_step: int = 1,
        weight_type: str = "exponential2",
    ):
        self.num_groups = num_groups
        self.num_states = num_states
        self.num_actions = num_actions

        # get data for a single machine
        self.costs = CostReward(num_states=num_states, rccc_wrt_max=rccc_wrt_max).costs
        self.rewards = -self.costs
        self.transitions = TransitionMatrix(
            num_states=num_states,
            prob_remain=prob_remain,
            deterioration_step=deterioration_step,
        ).transitions

        # generate data for multiple groups
        self.global_states = self.get_global_states()
        self.num_global_states = len(self.global_states)
        self.global_actions = self.get_global_actions()
        self.num_global_actions = len(self.global_actions)
        self.global_transitions = self.generate_global_transition_matrix()
        self.global_costs = self.generate_global_cost_matrix()
        self.global_rewards = -self.global_costs

        # generate weights for multiple groups
        self.weights = FairWeight(
            num_groups=self.num_groups, weight_type=weight_type
        ).weights

        # TODO: remove after testing the correctness of the code
        for state_idx in range(len(self.global_states)):
            state_vector = state_int_index_to_vector(
                state_int_index=state_idx,
                num_groups=self.num_groups,
                num_states=self.num_states,
            )
            for group_idx in range(self.num_groups):
                assert (
                    state_vector[group_idx] == self.global_states[state_idx][group_idx]
                )

    def get_global_states(self) -> np.array:
        """A helper function used to get all possible states: cartesian s^D.

        Example (3 states, 2 groups):
            [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]

        Returns:
            (`np.array`): all possible states
        """

        # Used to generate state cartesian product
        state_D_dim_temp = [list(range(self.num_states))] * self.num_groups
        tuple_list_s = list(itertools.product(*state_D_dim_temp))
        return np.array(tuple_list_s)

    def get_global_actions(self) -> np.array:
        """ A helper function used to get all possible actions: [Keep] + [replace_1, ..., replace_D].

        Example (2 groups, 0: not replace, 1: replace):
            [0, 0; 1, 0; 0, 1]

        Returns:
            (`np.array`): all possible actions
        """

        # Keep all groups ([0]*D)
        all_zeros = np.zeros(self.num_groups, dtype=int)
        # Replace the n-th group (diagonal[replace_1, ..., replace_D])
        replace_n = np.eye(self.num_groups, dtype=int)
        return np.vstack([all_zeros, replace_n])

    def generate_global_cost_matrix(self) -> np.array:
        """Generate the cost matrix R(s, a, d) at state s taking action a for group d.

        Returns:
            global_costs (`np.array`): cost matrix
        """

        global_costs = np.zeros(
            [self.num_global_states, self.num_global_actions, self.num_groups]
        )
        for s in range(self.num_global_states):
            for a in range(self.num_global_actions):
                for d in range(self.num_groups):
                    global_costs[s, a, d] = self.costs[
                        self.global_states[s, d], self.global_actions[a, d]
                    ]
        return global_costs

    def generate_global_transition_matrix(self) -> np.array:
        """Generate the transition matrix Pr(s, s', a) from state s to state s' taking action a.

        Returns:
            global_transitions (`np.array`): transition matrix
        """

        # initialize the transition matrix
        global_transitions = np.zeros(
            [self.num_global_states, self.num_global_states, self.num_global_actions]
        )
        for s in range(self.num_global_states):
            for a in range(self.num_global_actions):
                for next_s in range(self.num_global_states):
                    temp_trans_prob = 1
                    for d in range(self.num_groups):
                        temp_trans_prob *= self.transitions[
                            self.global_states[s, d],
                            self.global_states[next_s, d],
                            self.global_actions[a, d],
                        ]
                    global_transitions[s, next_s, a] = temp_trans_prob
                # TODO: remove after testing the correctness of the code
                assert np.sum(global_transitions[s, :, a]) - 1 < 1e-5
        return global_transitions


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
    """Define the Markov chain transition matrix for a single machine."""

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
