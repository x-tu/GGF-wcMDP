from datetime import datetime

import pandas as pd

from algorithms.q_learning import QAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params
from utils.plots import plot_figures

params.update({"num_episodes": 300, "len_episode": 100, "num_samples": 10})

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

learning_rates = [1 / 16, 1 / 8, 1 / 4, 1 / 2]
exploration_rates = [0.1, 0.2, 0.3]

all_rewards = {}
for ep in exploration_rates:
    for lr in learning_rates:
        agent = QAgent(
            env=env,
            learning_rate=lr,
            discount_factor=params.gamma,
            exploration_rate=ep,
            decaying_factor=params.ql.decaying_factor,
        )
        episode_rewards = agent.run(
            num_episodes=params.num_episodes,
            len_episode=params.len_episode,
            num_samples=params.num_samples,
        )
        unique_key_str = f"{ep}-{lr}"
        all_rewards[unique_key_str] = episode_rewards
# plot the results
plot_figures(ep_list=exploration_rates, lr_list=learning_rates, all_rewards=all_rewards)
# save the results
all_rewards_df = pd.DataFrame.from_dict(all_rewards)
time_stamp = str(datetime.now().strftime("%m_%d_%H_%M_%S"))
all_rewards_df.to_csv(f"results/all_rewards_{time_stamp}.csv")
