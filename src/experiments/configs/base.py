"""This script includes all base parameters."""

from utils.common import DotDict

params = DotDict(
    {
        "num_actions": 2,
        "num_states": 3,
        "num_groups": 3,
        "weight_coefficient": 2,
        "ggi": True,
    }
)
