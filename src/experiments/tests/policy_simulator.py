import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params

# Notice: modify the file path to the one containing policy
policy = pd.read_csv("../results/policy/1026/policy_dqnS_4.csv")
# drop the index column
policy = policy.iloc[:, 1:]

# update parameters
params.num_episodes = 400

env = MachineReplacement(
    num_arms=params.num_groups,
    num_states=params.num_states,
    rccc_wrt_max=params.rccc_wrt_max,
    prob_remain=params.prob_remain,
    mat_type=params.mat_type,
    weight_coefficient=params.weight_coefficient,
    num_steps=params.len_episode,
    ggi=params.ggi,
    encoding_int=True,
)

# size of the problem
state_size = policy.shape[0]
action_size = policy.shape[1]

episode_rewards = []
for ep in range(params.num_episodes):
    # run LP
    ep_rewards = []
    for n in range(params.num_samples):
        init_state = random.randint(0, state_size - 1)
        state = env.reset(initial_state=init_state)
        total_reward = 0
        for t in range(params.len_episode):
            action_prob = policy.iloc[state].tolist()
            action = random.choices(range(len(action_prob)), weights=action_prob, k=1)[
                0
            ]
            next_observation, reward, done, _ = env.step(action)
            total_reward += (1 - done) * params.gamma ** t * reward
            state = next_observation
        ep_rewards.append(total_reward)
    # get the expected rewards by averaging over samples, and then sort
    rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
    episode_rewards.append(np.dot(env.weights, rewards_sorted))
sns.lineplot(episode_rewards)
print(np.mean(episode_rewards))
plt.show()
