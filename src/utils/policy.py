""" This module contains the functions to calculate and simulate the state value function."""

import numpy as np
from tqdm import tqdm

from solver.dual_mdp import LPData
from utils.common import DotDict


def simulation_state_value(
    params: DotDict, policy: np.array, mrp_data: LPData, initial_state_prob: np.array
):
    """Main function to simulate the state value function from the policy.

    Args:
        params (`DotDict`): the parameters used for the simulation
        policy (`np.array`): the policy pi
        mrp_data (`LPData`): the MRP data
        initial_state_prob (`np.array`): the initial state distribution mu
    """

    # get the size of the problem
    state_size = policy.shape[0]
    # run LP simulation
    sample_rewards = []
    # used to record rewards for each time step
    sample_rewards_by_time, state_value_list = [], []
    for n in tqdm(range(params.num_samples)):
        state = np.random.choice(range(state_size), p=initial_state_prob)
        total_reward = [0] * params.num_groups
        # used to record rewards for each time step
        total_reward_by_time = []
        for t in range(params.len_episode):
            action_prob = policy.iloc[state].tolist()
            action = np.random.choice(range(len(action_prob)), p=action_prob)
            reward_lp = mrp_data.global_costs[state, action]
            next_observation = np.random.choice(
                range(state_size), p=mrp_data.global_transitions[state, :, action]
            )
            total_reward += params.gamma ** t * reward_lp
            state = next_observation
            total_reward_by_time.append(total_reward.copy())
        sample_rewards.append(total_reward)
        sample_rewards_by_time.append(total_reward_by_time)
    # get the expected rewards by averaging over samples, and then sort
    state_value = sorted(np.mean(sample_rewards, axis=0))

    for t in range(params.len_episode):
        state_value_list.append(
            sorted(np.mean(np.array(sample_rewards_by_time)[:, t, :], axis=0))
        )

    return state_value, state_value_list


def calculate_state_value(
    discount: int,
    initial_state_prob: np.array,
    policy: np.array,
    rewards: np.array,
    transition_prob: np.array,
    time_horizon: int,
):
    """Main function to calculate the state value function from the policy.

    V = sum_{t=0}^T mu^T (gamma * pi * P)^t * pi * r

    Args:
        discount (`int`): the discount factor gamma
        initial_state_prob (`np.array`): the initial state distribution mu
        policy (`np.array`): the policy pi
        rewards (`np.array`): the reward matrix r
        transition_prob (`np.array`): the transition probability matrix P
        time_horizon (`int`): the time horizon T
    """

    # calculate the state value function
    _, _, R = rewards.shape
    state_value = np.zeros(R)
    state_value_list = [state_value.copy()]

    # transform the policy matrix to block diagonal matrix of size S * SA
    policy_trans = transform_policy_matrix(policy)
    transition_prob_trans = reshape_transition_matrix(transition_prob)
    rewards_trans = reshape_reward_matrix(rewards)

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
            rewards_trans,
        )
        state_value_list.append(state_value.copy())
    return state_value, state_value_list


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
