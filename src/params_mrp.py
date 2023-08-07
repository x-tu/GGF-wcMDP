import numpy as np
from typing import List


# Define the cost function for each arm
class CostReward:

    def __init__(self,
                 num_states: int,
                 num_arms: int,
                 rccc_wrt_max=1.5):
        self.num_s = num_states
        self.num_a = num_arms
        self.costs = np.zeros([self.num_s, self.num_a, 2])
        self.costs[:, :, 0] = self.quadratic()
        self.costs[:, :, 1] = rccc_wrt_max * (self.num_s-1)**2 * np.ones([self.num_s, self.num_a])
        self.costs = self.costs / np.max(self.costs)
        self.rewards = 1 - self.costs

    def linear(self):
        costs = np.zeros([self.num_s, self.num_a])
        for n in range(self.num_a):
            costs[:, n] = np.linspace(0, self.num_s-1, num=self.num_s)
        return costs

    def constant(self, k):
        return

    def quadratic(self):
        costs = np.zeros([self.num_s, self.num_a])
        for a in range(self.num_a):
            for s in range(self.num_s):
                costs[s, a] = s**2
        return costs


# Define the Markov dynamics for each arm
class MarkovChain:

    def __init__(self,
                 num_states: int,
                 num_arms: int,
                 prob_remain,
                 mat_type=1):
        self.num_s = num_states
        self.num_a = num_arms
        self.transitions = np.zeros([self.num_s, self.num_s, self.num_a, 2])
        self.transitions[:, :, :, 0] = self.deterioration(prob_remain, mat_type)
        self.transitions[:, :, :, 1] = self.pure_reset()

    def pure_reset_pmf(self):
        pmf = np.concatenate([np.ones([1, 1]), np.zeros([1, self.num_s - 1])], 1)
        return pmf

    def pure_reset(self):
        transitions = np.zeros([self.num_s, self.num_s, self.num_a])
        for n in range(self.num_a):
            for s in range(self.num_s):
                transitions[s, :, n] = self.pure_reset_pmf()
        return transitions

    def deterioration(self, prob_remain, mat_type):
        transitions = np.zeros([self.num_s, self.num_s, self.num_a])
        for n in range(self.num_a):
            if mat_type == 1:
                for s in range(self.num_s-1):
                    transitions[s, s, n] = prob_remain[n]
                    transitions[s, s+1, n] = 1-prob_remain[n]
                transitions[self.num_s-1, self.num_s-1, n] = 1
            elif mat_type == 2:
                for s in range(self.num_s-2):
                    transitions[s, s, n] = prob_remain[n]
                    transitions[s, s+1, n] = (1-prob_remain[n])/2
                    transitions[s, s+2, n] = (1-prob_remain[n])/2
                transitions[self.num_s-2, self.num_s-2, n] = prob_remain[n]
                transitions[self.num_s-2, self.num_s-1, n] = 1-prob_remain[n]
                transitions[self.num_s-1, self.num_s-1, n] = 1
            elif mat_type == 3:
                for s in range(self.num_s-2):
                    transitions[s, s, n] = prob_remain[n]
                    transitions[s, s+1, n] = 2*(1-prob_remain[n])/3
                    transitions[s, s+2, n] = (1-prob_remain[n])/3
                transitions[self.num_s-2, self.num_s-2, n] = prob_remain[n]
                transitions[self.num_s-2, self.num_s-1, n] = 1-prob_remain[n]
                transitions[self.num_s-1, self.num_s-1, n] = 1
            elif mat_type == 4:
                for s in range(self.num_s-1):
                    transitions[s, s, n] = prob_remain[n]
                    transitions[s, s+1:self.num_s, n] = ((1-prob_remain[n])/(self.num_s-s))*np.ones([self.num_s-s])
                transitions[self.num_s-1, self.num_s-1, n] = 1
        return transitions


# Define the fairness weight for each arm
class FairWeight:
    def __init__(self, num_arms: int, weight_coefficient):
        if np.isscalar(weight_coefficient):
            self.weights = np.array([1 / (weight_coefficient ** i) for i in range(num_arms)])
            self.weights = self.weights / np.sum(self.weights)
        elif len(weight_coefficient) == num_arms:
            self.weights = weight_coefficient
            self.weights = self.weights / np.sum(self.weights)
        else:
            raise TypeError("`weight_coef` should be either scalar or array with length reward_space")


def get_state_list(num_states, num_arms):
    """ A helper function used to get state list: cartesian s^D.

    Example (3 states, 2 groups):
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    Returns:
        state_list: state tuple list

    """
    # generate state indices
    state_indices = np.arange(num_states)
    # get cartesian product
    state_indices_cartesian = np.meshgrid(
        *([state_indices] * num_arms), indexing="ij"
    )
    # reshape and convert to list
    state_list = (
        np.stack(state_indices_cartesian, axis=-1)
        .reshape(-1, num_arms)
        .tolist()
    )
    return state_list


# def state_decoding(num_states, num_arms, state_index):
#     sys_states = np.zeros(num_arms, dtype=int)
#     sys_states[num_arms-1] = np.mod(state_index+1, num_states)
#     sys_states[0] = (state_index-sys_states[num_arms-1]) // (num_states**(num_arms-1)) + 1
#     for ar in range(1, num_arms-1):
#         val = sys_states[num_arms-1]
#         for x in range(ar-1):
#             val = val + (sys_states[x]-1)*(num_states**(num_arms-x+1))
#         sys_states[ar] = (state_index-val) // (num_states**(num_arms-ar+1)) + 1
#     return sys_states
