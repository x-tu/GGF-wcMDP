""" This module contains the functions to calculate and simulate the state value function."""

import numpy as np
from scipy import stats
from tqdm import tqdm

from utils.common import DotDict
from utils.encoding import state_int_index_to_vector, state_vector_to_int_index
from utils.mrp import MRPData


def simulate_permuted_state_value(
    params: DotDict,
    policy: np.array,
    mrp_data: MRPData,
    initial_state_prob: np.array,
    all_permutation_matrix: np.array,
):
    """Simulate the state value function from the policy with permutation."""

    Vt = np.zeros((params.num_samples, params.num_groups, params.len_episode))
    for n in range(params.num_samples):
        # select the initial state according to the initial state distribution
        s_idx = np.random.choice(
            range(mrp_data.num_global_states), p=initial_state_prob
        )
        for t in range(params.len_episode):
            perm_matrix = all_permutation_matrix[
                np.random.choice(len(all_permutation_matrix))
            ]
            inverse_perm_matrix = np.linalg.inv(perm_matrix)
            # permute the state
            s_vector = state_int_index_to_vector(
                state_int_index=s_idx,
                num_groups=params.num_groups,
                num_states=params.num_states,
            )
            perm_s_vector = np.matmul(perm_matrix, s_vector)
            perm_s_idx = state_vector_to_int_index(
                state_vector=perm_s_vector, num_states=params.num_states
            )
            # select the permute action
            perm_a_prob = policy.iloc[perm_s_idx]
            perm_a_idx = np.random.choice(range(len(perm_a_prob)), p=perm_a_prob)
            # TODO: 1) permute the action back, 2) keep using permuted action, but permute the reward
            perm_reward = mrp_data.global_costs[perm_s_idx, perm_a_idx]
            reward = np.matmul(perm_reward, inverse_perm_matrix)
            Vt[n, :, t] = params.gamma**t * reward
            # transit to the next state
            perm_next_s_prob = mrp_data.global_transitions[perm_s_idx, :, perm_a_idx]
            perm_next_s_idx = np.random.choice(
                range(len(perm_next_s_prob)), p=perm_next_s_prob
            )
            perm_next_s = state_int_index_to_vector(
                perm_next_s_idx, params.num_groups, params.num_states
            )
            next_s = np.matmul(perm_next_s, inverse_perm_matrix)
            next_s_idx = state_vector_to_int_index(next_s, params.num_states)
            s_idx = next_s_idx
    return Vt


def simulation_state_value(
    params: DotDict, policy: np.array, mrp_data: MRPData, initial_state_prob: np.array
) -> np.array:
    """Main function to simulate the state value function from the policy.

    Args:
        params (`DotDict`): the parameters used for the simulation
        policy (`np.array`): the policy pi
        mrp_data (`MRPData`): the MRP data
        initial_state_prob (`np.array`): the initial state distribution mu

    Returns:
        sorted_expected_state_values (`np.array`):
            the sorted expected state value vector of size (T, N) for each time step and N groups
    """

    # get the size of the problem
    state_size = policy.shape[0]
    # initialize state values with matrix size of (T, N, K, S)
    state_values = np.zeros(
        (params.len_episode, params.num_groups, params.num_samples, state_size)
    )

    for n in tqdm(range(params.num_samples)):
        for init_state in range(state_size):
            if initial_state_prob[init_state] == 0:
                # no need to simulate if the initial state probability is 0
                state_values[:, :, n, init_state] = 0
            else:
                state = init_state
                total_reward = [0] * params.num_groups
                for t in range(params.len_episode):
                    action_prob = policy.iloc[state].tolist()
                    action = np.random.choice(range(len(action_prob)), p=action_prob)
                    reward_lp = mrp_data.global_costs[state, action]
                    next_observation = np.random.choice(
                        range(state_size),
                        p=mrp_data.global_transitions[state, :, action],
                    )
                    total_reward += params.gamma**t * reward_lp
                    state = next_observation
                    state_values[t, :, n, init_state] = total_reward.copy()
    # calculate the expected state values based on the initial state distribution
    weighted_state_values = np.matmul(state_values, initial_state_prob)
    # take averages over samples and sort the expected state values
    sorted_expected_state_values = np.sort(np.mean(weighted_state_values, axis=2))
    return sorted_expected_state_values


def calculate_state_value(
    discount: float,
    initial_state_prob: np.array,
    policy: np.array,
    reward_or_cost: np.array,
    transition_prob: np.array,
    time_horizon: int,
):
    """Main function to calculate the state value function from the policy.

    V = sum_{t=0}^T mu^T (gamma * pi * P)^t * pi * r

    Args:
        discount (`float`): the discount factor gamma
        initial_state_prob (`np.array`): the initial state distribution mu
        policy (`np.array`): the policy pi
        reward_or_cost (`np.array`): the reward or cost matrix r
        transition_prob (`np.array`): the transition probability matrix P
        time_horizon (`int`): the time horizon T
    """

    # calculate the state value function
    _, _, R = reward_or_cost.shape
    state_value = np.zeros(R)
    state_value_list = [state_value.copy()]

    # transform the policy matrix to block diagonal matrix of size S * SA
    policy_trans = transform_policy_matrix(policy)
    # reshape the transition and reward_or_cost matrix
    transition_prob_trans = reshape_transition_matrix(transition_prob)
    reward_or_cost_trans = reshape_reward_matrix(reward_or_cost)

    # calculate the state value function
    for t in range(time_horizon):
        state_value += np.matmul(
            np.matmul(
                np.matmul(
                    initial_state_prob,
                    np.linalg.matrix_power(
                        discount * np.matmul(policy_trans, transition_prob_trans), t
                    ),
                ),
                policy_trans,
            ),
            reward_or_cost_trans,
        )
        state_value_list.append(state_value.copy())
    return state_value_list


def calculate_visitation_freq(
    discount: float,
    initial_state_prob: np.array,
    policy: np.array,
    transition_prob: np.array,
    time_horizon: int,
):
    """Main function to calculate the state visitation frequency from the policy.

    X = sum_{t=0}^T mu^T (gamma * pi * P)^t * pi

    Args:
        discount (`int`): the discount factor gamma
        initial_state_prob (`np.array`): the initial state distribution mu
        policy (`np.array`): the policy pi
        transition_prob (`np.array`): the transition probability matrix P
        time_horizon (`int`): the time horizon T
    """

    # calculate the state value function
    _, S, A = transition_prob.shape
    visitation_freq = np.zeros(S * A)

    # transform the policy matrix to block diagonal matrix of size S * SA
    policy_trans = transform_policy_matrix(policy)
    # reshape the transition and reward matrix
    transition_prob_trans = reshape_transition_matrix(transition_prob)

    # calculate the state value function
    for t in range(time_horizon):
        visitation_freq += np.matmul(
            np.matmul(
                initial_state_prob,
                np.linalg.matrix_power(
                    discount * np.matmul(policy_trans, transition_prob_trans), t
                ),
            ),
            policy_trans,
        )
    return visitation_freq.reshape(S, A)


def transform_policy_matrix(policy_matrix: np.array) -> np.array:
    """A helper function to transform a matrix into a block diagonal matrix.

    Args:
        policy_matrix (`np.array`): the original matrix to be transformed.

    Returns:
        new_policy_matrix (`np.array`): the block diagonal matrix.
    """

    # get the shape of the matrix
    S, A = policy_matrix.shape
    new_policy_matrix = np.zeros((S, S * A))

    # fill in the block diagonal matrix
    for i in range(S):
        new_policy_matrix[i, i * A : (i + 1) * A] = policy_matrix[i, :]
    return new_policy_matrix


def reshape_transition_matrix(transition_matrix: np.array) -> np.array:
    """A helper function to reshape the transition matrix from size S,S',A to SA, S'.

    Args:
        transition_matrix (`np.array`): the original transition matrix.

    Returns:
        new_transition_matrix (`np.array`): the reshaped transition matrix.
    """

    S, S_prime, A = transition_matrix.shape
    new_transition_matrix = np.zeros((S * A, S_prime))
    for sp in range(S_prime):
        new_transition_matrix[:, sp] = transition_matrix[:, sp, :].reshape(S * A)
    return new_transition_matrix


def reshape_reward_matrix(reward_matrix: np.array) -> np.array:
    """A helper function to reshape the reward matrix from size S,A,r to SA, r.

    Args:
        reward_matrix (`np.array`): the original reward matrix.

    Returns:
        new_reward_matrix (`np.array`): the reshaped reward matrix.
    """

    S, A, R = reward_matrix.shape
    new_reward_matrix = np.zeros((S * A, R))
    for r in range(R):
        new_reward_matrix[:, r] = reward_matrix[:, :, r].reshape(S * A)
    return new_reward_matrix


def check_equal_means(groups):
    f_statistic, p_value = stats.f_oneway(*groups)
    return p_value > 0.05
