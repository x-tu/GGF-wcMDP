import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.dqn import DQNAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params

params.update({"num_episodes": 200, "len_episode": 100, "num_samples": 8})

env = MachineReplacement(
    num_arms=params.num_groups,
    num_states=params.num_states,
    rccc_wrt_max=params.rccc_wrt_max,
    prob_remain=params.prob_remain,
    mat_type=params.mat_type,
    weight_coefficient=params.weight_coefficient,
    num_steps=params.len_episode,
    ggi=params.ggi,
    encoding_int=False,
)

agent = DQNAgent(
    env=env,
    h_size=params.dqn.h_size,
    learning_rate=params.dqn.alpha,
    discount_factor=params.gamma,
    exploration_rate=params.dqn.epsilon,
    decaying_factor=params.dqn.decaying_factor,
    deterministic=True,
)
episode_rewards = agent.run(
    num_episodes=params.num_episodes,
    len_episode=params.len_episode,
    num_samples=params.num_samples,
)
sns.lineplot(episode_rewards)
plt.show()
