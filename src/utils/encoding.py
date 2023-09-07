import numpy as np


def state_vector_to_int_index(state_vector: np.array, num_states: int) -> int:
    """Convert a state vector to an integer index.

    Example: [0, 1, 2], 3 states -> 0 * 3^2 + 1 * 3^1 + 2 * 3^0 = 5

    Args:
        state_vector (np.array): A list of integers, each of which denotes the machine status.
        num_states (int): The number of states.

    Returns:
        state_int (int): An integer index.
    """

    state_int = 0
    num_arms = len(state_vector)
    for i in range(num_arms):
        state_int += state_vector[num_arms - 1 - i] * (num_states ** i)
    return int(state_int)


def state_int_index_to_vector(
    state_int_index: int, num_arms: int, num_states: int
) -> np.array:
    """Convert an integer index to a state vector.

    Example: 5 = 0 * 3^2 + 1 * 3^1 + 2 * 3^0 -> [0, 1, 2]

    Args:
        state_int_index (int): An integer index.
        num_arms (int): The number of arms.
        num_states (int): The number of states.

    Returns:
        state_vector (np.array): A numpy list of integers, each of which denotes the machine status.
    """

    state_vector = []
    for i in range(num_arms):
        state_vector.insert(0, state_int_index % num_states)
        state_int_index = state_int_index // num_states
    return np.array(state_vector)


def action_vector_to_int_index(action_vector) -> int:
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
