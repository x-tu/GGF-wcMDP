"""This script includes all base parameters."""
import numpy as np

from utils.common import DotDict

params = DotDict(
    {
        "num_actions": 2,
        "num_states": 3,
        "num_groups": 3,
        "weight_coefficient": 2,
        "ggi": True,
        "num_episodes": 500,
        "len_episode": 500,
        "num_samples": 5,
        "gamma": 0.95,
        "rccc_wrt_max": 0.5,
        "prob_remain": np.linspace(start=0.5, stop=0.9, num=3),
        "mat_type": 1,
        # parameters for the Q-learning
        "ql": DotDict(
            {
                "alpha": 0.15,  # learning rate
                "epsilon": 0.3,  # exploration rate
                "decaying_factor": 0.95,  # decaying factor for epsilon
            }
        ),
        # parameters for the DQN
        "dqn": DotDict(
            {
                "h_size": 64,  # hidden layer size
                "alpha": 1e-3,  # learning rate
                "epsilon": 1.0,  # exploration rate
                "decaying_factor": 0.99,  # decaying factor for epsilon
                "deterministic": False,  # whether to use deterministic policy
                "encoding_int": False,  # whether to encode the state as integer
            }
        ),
    }
)
