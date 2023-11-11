"""This module contains functions for encoding and decoding state and action vectors."""

import numpy as np


def state_vector_to_int_index(state_vector: np.array, num_states: int) -> int:
    """Convert a state vector to an integer index.

    Example: [0, 1, 2], 3 states -> 0 * 3^2 + 1 * 3^1 + 2 * 3^0 = 5

    Args:
        state_vector (np.array): a list of integers, each of which denotes the machine status.
        num_states (int): number of (base) states.

    Returns:
        state_int (int): integer state index.
    """

    state_int = 0
    num_groups = len(state_vector)
    for i in range(num_groups):
        state_int += state_vector[num_groups - 1 - i] * (num_states ** i)
    return int(state_int)


def state_int_index_to_vector(
    state_int_index: int, num_groups: int, num_states: int
) -> np.array:
    """Convert an integer index to a state vector.

    Example: 5 = 0 * 3^2 + 1 * 3^1 + 2 * 3^0 -> [0, 1, 2]

    Args:
        state_int_index (int): An integer index.
        num_groups (int): The number of groups.
        num_states (int): The number of states.

    Returns:
        state_vector (np.array): A numpy list of integers, each of which denotes the machine status.
    """

    state_vector = []
    for i in range(num_groups):
        state_vector.insert(0, state_int_index % num_states)
        state_int_index = state_int_index // num_states
    return np.array(state_vector)


def action_vector_to_int_index(action_vector: list) -> int:
    """Convert an action vector to an integer index.

    Example: [0, 1, 0] -> 2

    Args:
        action_vector

    Returns:
        action_int (int): An integer index.
    """
    action_int = 0
    for i in action_vector:
        if i > 0:
            action_int = i
            break
    return int(action_int)
