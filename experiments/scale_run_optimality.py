import matplotlib.pyplot as plt
import pandas as pd

from experiments.batch_run_optimality import train_agent
from experiments.configs.base import params
from stable_baselines3 import A2C, PPO, SAC, TD3
from utils.plots import moving_average

params.update({"num_episodes": 1000, "len_episode": 300})
FILE_OUT = False

model, training_rewards = train_agent(PPO, params)
file_name = f"experiments/tmp/learning_reward_ppo_{params.identifier}.csv"
if FILE_OUT:
    pd.DataFrame(training_rewards).to_csv(file_name)
plt.plot(moving_average(training_rewards, window=10), label="PPO")

print("============= plotting")
# plot figures
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("GGF Expected Returns")
plt.title("Learning Curves (Smoothed)")
plt.savefig(f"experiments/tmp/learning_reward_{params.identifier}.png")
plt.show()
