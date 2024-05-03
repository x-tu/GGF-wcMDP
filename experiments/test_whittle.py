import numpy as np

from algorithms.whittle import Whittle
from experiments.configs.base import params
from utils.count import CountMDP
from utils.mrp import MRPData

params.num_states = 3
params.num_groups = 3

# count_mdp = CountMDP(num_groups=2, num_resource=1, num_states=3)
mrp = MRPData(num_groups=params.num_groups, num_states=params.num_states)
costs = mrp.costs[:, :, 0].T
transitions = np.zeros((params.num_states, params.num_states, 2, params.num_groups))
for arm in range(2):
    transitions[:, :, :, arm] = mrp.transitions[arm, :, :, :]

whittle_agent = Whittle(
    num_states=params.num_states,
    num_arms=params.num_groups,
    reward=costs,
    transition=transitions,
    horizon=300,
)
whittle_agent.whittle_brute_force(0, 1, 1000)
whittle_agent.Whittle_policy(
    whittle_indices=whittle_agent.w_indices, n_selection=1, current_x=1, current_t=1
)
print(whittle_agent.w_indices)
