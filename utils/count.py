from typing import Union

import numpy as np

from utils.encoding import state_int_index_to_vector, state_vector_to_int_index
from utils.mrp import MRPData


def count_to_normal(count_representation):
    normal_representation = []
    for index, count in enumerate(count_representation):
        normal_representation.extend([index] * int(count))
    return normal_representation


class CountMDP(MRPData):
    def __init__(
        self,
        num_groups: int = 2,
        num_states: int = 3,
        num_actions: int = 2,
        num_resource: int = 1,
        rccc_wrt_max: float = 1.5,
        prob_remain: Union[float, list] = 0.8,
        deterioration_step: int = 1,
        cost_types_operation: Union[list, str] = "quadratic",
        cost_types_replace: Union[list, str] = "rccc",
    ):
        super().__init__(
            num_groups=num_groups,
            num_states=num_states,
            num_actions=num_actions,
            rccc_wrt_max=rccc_wrt_max,
            prob_remain=prob_remain,
            deterioration_step=deterioration_step,
            cost_types_operation=cost_types_operation,
            cost_types_replace=cost_types_replace,
        )

        self.global_states_list = []
        self.get_global_count_states(
            current_combination=[],
            remaining_sum=self.num_groups,
            remaining_states=self.num_states,
        )
        # make sure the states are uniquely ordered
        self.count_states = sorted(self.global_states_list, reverse=True)
        self.count_state_props = [
            np.array(cs + [num_resource]) / num_groups for cs in self.count_states
        ]
        self.num_count_states = len(self.count_states)
        self.count_actions = self.get_global_count_actions()
        self.count_action_props = [
            np.array(ca) / num_groups for ca in self.count_actions
        ]
        self.num_count_actions = len(self.count_actions)
        self.count_transitions = self.get_global_count_transitions()
        self.count_costs = self.get_global_count_costs()
        self.count_rewards = -self.count_costs

        # mapping for quick access
        self.mapping_x_to_idx = {
            str(state): idx for idx, state in enumerate(self.count_states)
        }
        self.x_to_s_idx_mapping = {
            str(state): state_vector_to_int_index(
                count_to_normal(state), self.num_states
            )
            for state in self.count_states
        }
        self.s_idx_to_x_mapping = {
            str(value): eval(key) for key, value in self.x_to_s_idx_mapping.items()
        }
        self.sc_idx_to_s_idx_mapping, sc_idx = {}, 0
        for value in sorted(self.x_to_s_idx_mapping.values()):
            self.sc_idx_to_s_idx_mapping[sc_idx] = value
            sc_idx += 1
        self.ac_to_idx_mapping = {
            str(action): idx for idx, action in enumerate(self.count_actions)
        }
        self.count_env_idx_mapping = self.get_env_mapping()

    def get_env_mapping(self):
        count_env_idx_mapping = {}
        for s_idx in range(len(self.count_states)):
            s_count = self.count_states[s_idx]
            s_vec = count_to_normal(s_count)
            for action in range(len(self.global_actions)):
                a_vec = [0] * self.num_groups
                if action > 0:
                    a_vec[action - 1] = 1
                ac_vec = np.array(self.action_normal_to_count(a_vec, s_vec))
                a_idx = self.ac_to_idx_mapping[str(ac_vec)]
                count_env_idx_mapping[(s_idx, action)] = a_idx
        return count_env_idx_mapping

    def get_global_count_states(
        self, current_combination, remaining_sum, remaining_states
    ):
        if remaining_states == 0:
            if remaining_sum == 0:
                self.global_states_list.append(current_combination)
            return

        for i in range(remaining_sum + 1):
            new_combination = current_combination + [i]
            self.get_global_count_states(
                new_combination, remaining_sum - i, remaining_states - 1
            )

    def get_global_count_actions(self) -> np.array:
        # Keep all groups ([0]*D)
        all_zeros = np.zeros(self.num_states, dtype=int)
        # Replace the n-th group (diagonal[replace_1, ..., replace_D])
        replace_n = np.eye(self.num_states, dtype=int)
        return np.vstack([all_zeros, replace_n])

    def get_global_count_transitions(self) -> np.array:
        global_transitions = np.zeros(
            (self.num_count_states, self.num_count_states, self.num_count_actions)
        )
        for sc_idx in range(self.num_count_states):
            s_count = self.count_states[sc_idx]
            s_vec = count_to_normal(s_count)
            s_idx = state_vector_to_int_index(s_vec, self.num_states)
            for next_s_idx in range(self.num_global_states):
                next_s_vec = state_int_index_to_vector(
                    next_s_idx, self.num_groups, self.num_states
                )
                next_s_count = self.normal_to_count(next_s_vec)
                next_sc_idx = self.count_states.index(next_s_count)
                for a_idx in range(len(self.global_actions)):
                    a_vec = self.global_actions[a_idx].tolist()
                    ac_vec = self.action_normal_to_count(a_vec, s_vec)
                    ac_idx = self.count_actions.tolist().index(ac_vec)
                    global_transitions[
                        sc_idx, next_sc_idx, ac_idx
                    ] += self.global_transitions[s_idx, next_s_idx, a_idx]
        # normalize the transition matrix
        sum_transitions = np.sum(global_transitions, axis=1)[:, np.newaxis]
        # avoid division by zero
        sum_transitions[sum_transitions == 0] = 1
        global_transitions /= sum_transitions
        return global_transitions

    def action_normal_to_count(self, a_vec, s_vec):
        ac_vec = [0] * self.num_states
        for idx in range(self.num_groups):
            if a_vec[idx] > 0:
                ac_vec[s_vec[idx]] = a_vec[idx]
        return ac_vec

    def normal_to_count(self, normal_representation):
        count_representation = [0] * self.num_states
        for element in normal_representation:
            count_representation[element] += 1
        return count_representation

    def get_global_count_costs(self) -> np.array:
        global_costs = np.zeros((self.num_count_states, self.num_count_actions))
        for sc_idx in range(self.num_count_states):
            s_count = self.count_states[sc_idx]
            for ac_idx in range(self.num_count_actions):
                reward = 0
                a_count = self.count_actions[ac_idx]
                if all(s_count[i] >= a_count[i] for i in range(self.num_states)):
                    for idx in range(self.num_states):
                        indicator = 1 if s_count[idx] > 0 else 0
                        replace_cost = a_count[idx] * self.costs[0, idx, 1]
                        do_nothing_cost = (s_count[idx] - a_count[idx]) * self.costs[
                            0, idx, 0
                        ]
                        reward += indicator * (replace_cost + do_nothing_cost)
                    global_costs[sc_idx, ac_idx] = reward
                else:
                    global_costs[sc_idx, ac_idx] = 1e6
        return global_costs


def softmax(x):
    # Subtracting the maximum value for numerical stability
    exp_x = np.exp(x - np.max(x)).round(4)
    prob = exp_x / exp_x.sum()
    return prob
