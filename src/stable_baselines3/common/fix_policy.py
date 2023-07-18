""" This module contains the functions used for solving fixed policy by the LP solver."""


def policy_convertor(policy_nn) -> dict:
    """ Convert the policy network to policy acceptable by the LP solver.

    Args:
        policy_nn: policy network

    Returns:
        state_action_pair: a dictionary of state-action pair
    """

    state_dim = policy_nn.observation_space.n
    state_action_pair = {}
    for state in range(state_dim):
        state_action_pair[state] = int(policy_nn.predict(state)[0])
    return state_action_pair


def policy_convertor_from_q_values(model) -> dict:
    """ Convert the policy network to policy acceptable by the LP solver.

    Args:
        policy_nn: policy network

    Returns:
        state_action_pair: a dictionary of state-action pair
    """
    state_action_pair = {}
    state_dim = model.observation_space.n
    for state in range(state_dim):
        state_action_pair[state] = int(model.predict(state)[0])
    return state_action_pair
