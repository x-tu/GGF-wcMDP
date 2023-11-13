"""This script includes all base parameters."""

from utils.common import DotDict

params = DotDict(
    {
        "num_actions": 2,
        "num_states": 3,
        "num_groups": 3,
        "ggi": True,
        "num_episodes": 1000,
        "len_episode": 100,
        "num_samples": 10,
        "gamma": 0.95,
        "rccc_wrt_max": 0.5,
        "mat_type": 1,
        "prob_remain": 0.8,
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
        # hyperparameters for the A2C
        "a2c": DotDict(
            {
                "alpha": 0.001,
                "policy_loss_weight": 1.0,
                "value_loss_weight": 0.6,
                "entropy_loss_weight": 0.001,
                "max_n_steps": 10,
                "n_workers": 8,
                "tau": 0.95,
                "encoding_int": False,  # whether to encode the state as integer
                "hidden_dims": (64, 64),
            }
        ),
    }
)
