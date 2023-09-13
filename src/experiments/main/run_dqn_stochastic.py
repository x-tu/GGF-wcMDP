import numpy as np

from algorithms.dqn_stochastic import StochasticDQNAgent
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

agent = StochasticDQNAgent(
    env=env, discount=0.95, ggi_flag=env.ggi, weights=env.weights
)

agent.train(num_episodes=100, len_episode=100, num_samples=10, deterministic=False)
