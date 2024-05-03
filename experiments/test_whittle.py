import numpy as np

from algorithms.whittle import Whittle
from experiments.configs.base import params
from utils.count import CountMDP
from utils.mrp import MRPData

params.num_states = 3
params.num_groups = 2

# count_mdp = CountMDP(num_groups=2, num_resource=1, num_states=3)
mrp = MRPData(num_groups=params.num_groups, num_states=params.num_states)

whittle_agent = Whittle(
    num_states=params.num_states,
    num_arms=params.num_groups,
    reward=mrp.costs,
    transition=mrp.transitions,
    horizon=300,
)
whittle_agent.whittle_brute_force(lower_bound=0, upper_bound=1, num_trials=1000)

# MC simulation to evaluate the policy
state = np.array([0, 0])
total_reward = 0
for t in range(300):
    # with 1 arm to be selected
    action = whittle_agent.Whittle_policy(
        whittle_indices=whittle_agent.w_indices,
        n_selection=1,
        current_x=state,
        current_t=t,
    )
    # get the reward
    next_state = np.zeros_like(state)
    for arm in range(params.num_groups):
        total_reward += (mrp.costs[arm, state[arm], action[arm]]) * (params.gamma**t)
        next_state[arm] = np.random.choice(
            range(params.num_states), p=mrp.transitions[arm, state[arm], :, action[arm]]
        )
    state = next_state
print(
    total_reward / params.num_groups
)  # to get the mean group reward to compare with weighted average comparison
