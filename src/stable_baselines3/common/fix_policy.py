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
