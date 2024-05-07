import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.whittle import Whittle
from experiments.configs.base import params
from utils.mrp import MRPData
from utils.policy import check_equal_means

params.num_states = 3
params.num_groups = 2
params.len_episode = 300
budget = 1
n_runs = 1000

mrp = MRPData(num_groups=params.num_groups, num_states=params.num_states)

whittle_agent = Whittle(
    num_states=params.num_states,
    num_arms=params.num_groups,
    reward=mrp.rewards + 1,  # (arm, x, act)
    transition=mrp.transitions,  # (arm, x, x', act)
    horizon=params.len_episode,
)
whittle_agent.whittle_brute_force(lower_bound=-1.1, upper_bound=0.1, num_trials=1000)

# MC simulation to evaluate the policy
group_rewards = np.zeros((params.num_groups, n_runs))
for run in range(n_runs):
    s_idx = np.random.choice(len(mrp.global_states))
    state = mrp.global_states[s_idx]
    for t in range(params.len_episode):
        action = whittle_agent.Whittle_policy(
            whittle_indices=whittle_agent.w_indices,
            n_selection=budget,
            current_x=state,
            current_t=t,
        )
        # get the reward
        next_state = np.zeros_like(state)
        for arm in range(params.num_groups):
            group_rewards[arm, run] += (
                mrp.rewards[arm, state[arm], action[arm]] + 1
            ) * (params.gamma**t)
            next_state[arm] = np.random.choice(
                range(params.num_states),
                p=mrp.transitions[arm, state[arm], :, action[arm]],
            )
        state = next_state
# to get the mean group reward to compare with weighted average comparison
total_rewards = group_rewards.sum(axis=0) / params.num_groups
# mean rewards
print("Mean: ", total_rewards.mean())
print("Std:", total_rewards.std() / np.sqrt(n_runs))

print(check_equal_means(group_rewards))

rewards_df = pd.DataFrame(group_rewards.T)
rewards_df.columns = [f"Group {i}" for i in range(params.num_groups)]
# plot the mean and variance of the two groups
rewards_df.mean().plot(kind="bar", yerr=rewards_df.std())
# label is horizontal
plt.xticks(rotation=0)
plt.show()

# save the rewards
rewards_df.to_csv(
    f"experiments/tmp/rewards_whittle{params.num_groups}.csv", index=False
)
