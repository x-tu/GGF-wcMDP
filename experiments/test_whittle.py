import numpy as np
import pandas as pd
from scipy import stats

from algorithms.whittle import Whittle
from experiments.configs.base import params
from utils.mrp import MRPData

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


def check_equal_means(groups):
    f_statistic, p_value = stats.f_oneway(*groups)
    return p_value > 0.05


print(check_equal_means(group_rewards))

# plot histogram of the individual rewards
import matplotlib.pyplot as plt

rewards_df = pd.DataFrame(group_rewards.T)
rewards_df.columns = [f"Group {i}" for i in range(params.num_groups)]
# plot the mean and variance of the two groups
rewards_df.mean().plot(kind="bar", yerr=rewards_df.std())
# label is horizontal
plt.xticks(rotation=0)
plt.show()
# plot the distribution of the differences of rewards
differences = group_rewards[0] - group_rewards[1]
# plot the histogram of the differences with density estimation
plt.hist(differences, bins=50, alpha=0.5, density=True)
# plt.bar(differences, stats.norm.pdf(differences, np.mean(differences), np.std(differences)))
plt.show()

# save the rewards
rewards_df.to_csv("experiments/tmp/whittle_rewards.csv", index=False)
