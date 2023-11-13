import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.dqn import DQNAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params

params.update(
    {"num_episodes": 200, "len_episode": 10, "num_samples": 5, "deterministic": True}
)

env = MachineReplacement(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    num_steps=params.len_episode,
    encoding_int=params.dqn.encoding_int,
)

agent = DQNAgent(
    env=env,
    h_size=params.dqn.h_size,
    learning_rate=params.dqn.alpha,
    discount_factor=params.gamma,
    exploration_rate=params.dqn.epsilon,
    decaying_factor=params.dqn.decaying_factor,
    deterministic=params.dqn.deterministic,
)
episode_rewards = agent.run(
    num_episodes=params.num_episodes,
    len_episode=params.len_episode,
    num_samples=params.num_samples,
)
sns.lineplot(episode_rewards)
plt.show()
