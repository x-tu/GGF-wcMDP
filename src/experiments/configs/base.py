"""This script includes all base parameters."""

from utils.common import DotDict

params = DotDict(
    {
        # general parameters for the MDP
        "num_actions": 2,
        "num_states": 3,
        "num_groups": 3,
        "ggi": True,
        "rccc_wrt_max": 0.5,
        "mat_type": 1,
        "weight_type": "exponential2",
        "cost_type_operation": "quadratic",
        "cost_type_replace": "rccc",
        "prob_remain": 0.8,
        "num_opt_solutions": 1,
        "num_episodes": 1000,
        "len_episode": 100,
        "num_samples": 10,
        "gamma": 0.95,
        "seed": 0,
        "prob1_state_idx": None,
        "deterministic_policy": False,
        # hyperparameters for the Q-learning
        "ql": DotDict(
            {
                "alpha": 0.3,  # learning rate
                "epsilon": 0.7,  # exploration rate
                "decaying_factor": 0.95,  # decaying factor for epsilon
                "deterministic": False,  # whether to use deterministic policy
            }
        ),
        # hyperparameters for the DQN
        "dqn": DotDict(
            {
                "h_size": 64,  # hidden layer size
                "alpha": 1e-4,  # learning rate
                "epsilon": 0.6,  # exploration rate
                "decaying_factor": 0.95,  # decaying factor for epsilon
                "deterministic": False,  # whether to use deterministic policy
                "encoding_int": False,  # whether to encode the state as integer
            }
        ),
    }
)
