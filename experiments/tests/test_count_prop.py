import numpy as np
import pandas as pd

from env.mrp_prop_count import PropCountMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, DDPG, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from utils.callbacks import SaveOnBestTrainingRewardCallback
from utils.plots import moving_average, plot_results

params.update({"num_episodes": 600, "len_episode": 300})
num_runs = 1
# Create log dir
log_dir = "experiments/tmp/"

all_rewards = np.zeros((params.num_episodes, num_runs))
for n in range(num_runs):
    env = PropCountMDPEnv(
        machine_range=[2, 2],
        resource_range=[1, 1],
        num_states=params.num_states,
        len_episode=params.len_episode,
    )
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=params.len_episode * 20, log_dir=log_dir
    )
    total_timesteps = params.num_episodes * params.len_episode

    model = DDPG(policy="MlpPolicy", env=env, learning_rate=1e-4, gamma=params.gamma)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    all_rewards[:, n] = env.env.training_rewards[: params.num_episodes]

model.save("experiments/tmp/ddpg1")
# save all rewards
df = pd.DataFrame(all_rewards)
df.to_csv("experiments/tmp/ddpg1.csv")

plot_results(log_dir)
