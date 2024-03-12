import numpy as np

from utils.mrp import MRPData
from utils.encoding import state_int_index_to_vector


def count_to_normal(count_representation):
    normal_representation = []
    for index, count in enumerate(count_representation):
        normal_representation.extend([index] * count)
    return normal_representation


class CountMDP(MRPData):
    def __init__(
            self,
            num_groups: int = 2,
            num_states: int = 3,
            num_actions: int = 2
    ):
        super().__init__(num_groups, num_states, num_actions)

        self.global_states_list = []
        self.get_global_count_states(current_combination=[],
                                     remaining_sum=self.num_groups,
                                     remaining_states=self.num_states)
        # make sure the states are uniquely ordered
        self.count_states = sorted(self.global_states_list, reverse=True)
        self.num_count_states = len(self.count_states)
        self.count_actions = self.get_global_count_actions()
        self.num_count_actions = len(self.count_actions)
        self.count_transitions = self.get_global_count_transitions()
        self.count_costs = self.get_global_count_costs()
        self.count_rewards = - self.count_costs

    def get_global_count_states(self, current_combination, remaining_sum, remaining_states):
        if remaining_states == 0:
            if remaining_sum == 0:
                self.global_states_list.append(current_combination)
            return

        for i in range(remaining_sum + 1):
            new_combination = current_combination + [i]
            self.get_global_count_states(new_combination, remaining_sum - i, remaining_states - 1)

    def get_global_count_actions(self) -> np.array:
        # Keep all groups ([0]*D)
        all_zeros = np.zeros(self.num_states, dtype=int)
        # Replace the n-th group (diagonal[replace_1, ..., replace_D])
        replace_n = np.eye(self.num_states, dtype=int)
        return np.vstack([all_zeros, replace_n])

    def get_global_count_transitions(self) -> np.array:
        global_transitions = np.zeros((self.num_count_states,
                                       self.num_count_states,
                                       self.num_count_actions))
        for s_idx in range(self.num_global_states):
            s_vec = state_int_index_to_vector(s_idx, self.num_groups, self.num_states)
            s_count = self.normal_to_count(s_vec)
            sc_idx = self.count_states.index(s_count)
            for next_s_idx in range(self.num_global_states):
                next_s_vec = state_int_index_to_vector(next_s_idx, self.num_groups, self.num_states)
                next_s_count = self.normal_to_count(next_s_vec)
                next_sc_idx = self.count_states.index(next_s_count)
                for ac_idx in range(self.num_count_actions):
                    if ac_idx != 0 and s_count[ac_idx - 1] > 0:
                        a_idx = 1
                    else:
                        a_idx = 0
                    global_transitions[sc_idx, next_sc_idx, ac_idx] += self.global_transitions[s_idx, next_s_idx, a_idx]
        # normalize the transition matrix
        global_transitions /= np.sum(global_transitions, axis=1)[:, np.newaxis]
        return global_transitions

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
                        do_nothing_cost = (s_count[idx] - a_count[idx]) * self.costs[0, idx, 0]
                        reward += indicator * (replace_cost + do_nothing_cost)
                    global_costs[sc_idx, ac_idx] = reward
                else:
                    global_costs[sc_idx, ac_idx] = 1e6
        return global_costs
