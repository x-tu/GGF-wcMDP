from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from algorithms.dqn_stochastic import DQNAgent
from env.mrp_env_rccc import MachineReplacement

# TODO: add observation space and action space
env = MachineReplacement(
    num_arms=3,
    num_states=3,
    rccc_wrt_max=0.5,
    prob_remain=np.linspace(start=0.5, stop=0.9, num=3),
    mat_type=1,
    weight_coefficient=2,
    num_steps=100,
    ggi=True,
    encoding_int=False,
    out_csv_name="test",
)

stochastic_agent = DQNAgent(
    env=env, discount=0.95, ggi_flag=env.ggi, weights=env.weights
)
stochastic_agent.train(
    num_episodes=200, len_episode=100, num_samples=10, deterministic=False
)

deterministic_agent = DQNAgent(
    env=env, discount=0.95, ggi_flag=env.ggi, weights=env.weights
)
deterministic_agent.train(
    num_episodes=200, len_episode=100, num_samples=10, deterministic=True
)

# plot the results
sns.lineplot(stochastic_agent.episode_rewards, label="Stochastic")
sns.lineplot(deterministic_agent.episode_rewards, label="Deterministic")
plt.xlabel("Episodes")
plt.ylabel("Discounted Reward")
plt.title("Learning Curve")
plt.show()
plt.savefig(
    "results/ep_rewards_" + str(datetime.now().strftime("%m_%d_%H_%M_%S")) + ".png"
)
sns.lineplot()
