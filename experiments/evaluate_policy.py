import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from env.mrp_simulation import PropCountSimMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from utils.count import CountMDP
from utils.policy import check_equal_means

params.num_groups = 2
num_resource = 1
algorithm = PPO
params.len_episode = 300
runs = 1000

# identifier
identifier = f"2_2_2_{num_resource}_{num_resource}"

env = PropCountSimMDPEnv(
    machine_range=[params.num_groups, params.num_groups],
    resource_range=[num_resource, num_resource],
    num_states=params.num_states,
    len_episode=params.len_episode,
)

env.num_budget = num_resource
count_mdp = CountMDP(
    num_groups=params.num_groups,
    num_resource=num_resource,
    num_states=params.num_states,
)
model = algorithm.load(f"experiments/tmp/{algorithm.__name__.lower()}_{identifier}")

identifier = f"2_2_{num_resource}_{num_resource}"
# Print the policy
for state in count_mdp.count_state_props:
    th_obs = torch.as_tensor(state).unsqueeze(0)
    env.observations = state
    action_priority, _ = model.predict(state, deterministic=False)
    count_action = env.select_action_by_priority(action_priority)
    print(
        "state",
        state * params.num_groups,
        ": ",
        count_action,
        " | ",
        action_priority.round(2),
    )


def simulate_group_rewards(runs):
    group_rewards = np.zeros((runs, params.num_groups))
    for run in tqdm(range(runs)):
        state = env.reset()
        for t in range(params.len_episode):
            # get the action
            action_priority, _ = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action_priority)
            state = next_state
        group_rewards[run, :] = env.last_group_rewards
    return group_rewards


# Evaluate the policy
group_rewards = simulate_group_rewards(runs=runs)
print(group_rewards.mean(axis=0))
rewards_df = pd.DataFrame(group_rewards)
rewards_df.columns = [f"Group {i+1}" for i in range(params.num_groups)]
rewards_df.to_csv(
    f"experiments/tmp/rewards_{algorithm.__name__.lower()}_{identifier}.csv",
    index=False,
)

print(check_equal_means(groups=group_rewards.T))
# plot the mean and variance of the two groups
rewards_df.mean().plot(kind="bar", yerr=rewards_df.std())
# specify the y range and interval
plt.ylim([0, 20])
# label is horizontal
plt.xticks(rotation=0)
plt.show()
