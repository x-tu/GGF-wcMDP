import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.q_learning import QAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params

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

agent = QAgent(
    env=env,
    learning_rate=params.ql.alpha,
    discount_factor=params.gamma,
    exploration_rate=params.ql.epsilon,
    decaying_factor=params.ql.decaying_factor,
)
episode_rewards = agent.run(
    num_episodes=params.num_episodes,
    len_episode=params.len_episode,
    num_samples=params.num_samples,
)
sns.lineplot(episode_rewards)
plt.show()
