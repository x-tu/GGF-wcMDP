import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from algorithms.q_learning_count import QAgent
from env.mrp_count import CountMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from utils.callbacks import SaveOnBestTrainingRewardCallback
from utils.count import CountMDP, count_to_normal
from utils.mrp import MRPData
from utils.plots import moving_average, plot_results

params.update({"num_episodes": 400, "len_episode": 300})
num_runs = 1
# Create log dir
log_dir = "../tmp/"

all_rewards = np.zeros((params.num_episodes, num_runs))
for n in range(num_runs):
    env = CountMDPEnv(
        num_groups=params.num_groups,
        num_states=params.num_states,
        num_actions=params.num_actions,
        len_episode=params.len_episode,
    )
    env = Monitor(env, log_dir)

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=params.len_episode * 20, log_dir=log_dir
    )
    total_timesteps = params.num_episodes * params.len_episode

    model = DQN(policy="MlpPolicy", env=env, learning_rate=1e-4, gamma=params.gamma)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    all_rewards[:, n] = env.env.training_rewards

model.save("dqn_mrp")
# save all rewards
df = pd.DataFrame(all_rewards)
df.to_csv("dqn33.csv")

# Print the policy
# for state in range(len(env.count_mdp.count_states)):
#     th_obs = torch.as_tensor(state).unsqueeze(0)
#     probs = model.policy.get_distribution(th_obs).distribution.probs[0].detach().numpy().round(4)
#     print("state", state, ": ", probs)

plot_results(log_dir)
