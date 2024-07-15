import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from algorithms.whittle import Whittle
from experiments.configs.base import params
from utils.mrp import MRPData
from utils.policy import check_equal_means

n_runs = 1000

mrp = MRPData(
    num_groups=params.num_groups,
    num_states=params.num_states,
    cost_types_operation=params.cost_type_operation,
    cost_types_replace=params.cost_type_replace,
)

file_name = f"experiments/tmp/rewards_whittle_{params.identifier}.csv"
policy_file_name = f"experiments/tmp/policy_whittle_{params.identifier}.csv"
whittle_index_file_name = f"experiments/tmp/whittle_index_{params.identifier}.csv"
whittle_agent = Whittle(
    num_states=params.num_states,
    num_arms=params.num_groups,
    reward=mrp.rewards + 1,  # (arm, x, act)
    transition=mrp.transitions,  # (arm, x, x', act)
    horizon=params.len_episode,
)
whittle_agent.whittle_brute_force(lower_bound=-2, upper_bound=2, num_trials=1000)

# MC simulation to evaluate the policy
group_rewards = np.zeros((params.num_groups, n_runs))
for run in tqdm(range(n_runs)):
    s_idx = np.random.choice(len(mrp.global_states))
    state = mrp.global_states[s_idx]
    for t in range(params.len_episode):
        action = whittle_agent.Whittle_policy(
            whittle_indices=whittle_agent.w_indices,
            n_selection=params.budget,
            current_x=state,
            current_t=t,
            shuffle_indices=params.ggi,
            force_to_use_all_resources=params.force_to_use_all_resources,
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
        # print(state, action, group_rewards[:, run], next_state)
        state = next_state
# to get the mean group reward to compare with weighted average comparison
total_rewards = group_rewards.sum(axis=0) / params.num_groups

rewards_df = pd.DataFrame(group_rewards.T)
rewards_df.columns = [f"Machine {i + 1}" for i in range(params.num_groups)]
# plot the mean and variance of the two groups
rewards_df.mean().plot(kind="bar", yerr=rewards_df.std())
print("Mean:", rewards_df.mean().values.round(params.digit))
print("Std:", rewards_df.std().values.round(params.digit))
print(check_equal_means(group_rewards))
# calculate GGF
print(
    "GGF: ", np.dot(np.sort(rewards_df.mean().values), np.array(mrp.weights)).round(4)
)

# label is horizontal
plt.xticks(rotation=0)
plt.show()


def get_policy(whittle_agent, params):
    from utils.count import CountMDP, count_to_normal

    count_mdp = CountMDP(
        num_groups=params.num_groups,
        num_states=params.num_states,
        num_resource=params.budget,
    )

    policy = {}
    # Print the policy
    for state in count_mdp.count_state_props:
        count_state = (state * whittle_agent.num_a)[: whittle_agent.num_x]
        # convert count state to state
        normal_state = count_to_normal(count_state)
        normal_action = whittle_agent.Whittle_policy(
            whittle_indices=whittle_agent.w_indices,
            n_selection=params.budget,
            current_x=normal_state,
            current_t=0,
            shuffle_indices=params.ggi,
            force_to_use_all_resources=params.force_to_use_all_resources,
        )
        # count action
        count_action = count_mdp.action_normal_to_count(normal_action, normal_state)
        policy[str(count_state)] = count_action
        print(count_state, count_action)
    return policy


# save the rewards
rewards_df.to_csv(file_name, index=False)
# transpose the dictionary to save the policy
pd.DataFrame.from_dict(get_policy(whittle_agent, params), orient="index").to_csv(
    policy_file_name
)
# save the whittle index
pd.DataFrame(whittle_agent.w_indices[0]).to_csv(whittle_index_file_name)
