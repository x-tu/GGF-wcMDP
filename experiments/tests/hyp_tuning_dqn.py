"""This script is used to test the hyperparameter tuning for DQN."""

import json
import logging
from datetime import datetime

import pandas as pd

from algorithms.dqn import DQNAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params
from utils.plots import plot_figures

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

params.update({"num_episodes": 300, "len_episode": 150, "num_samples": 15})

# batch-running parameters
GROUPS = [6]
DETERMINISTIC = [False]
LEARNING_RATES = [1e-4]
EXPLORATION_RATES = [0.7]
PLOT = True
SAVE_RESULTS = True

# for batching the experiments
for grp in GROUPS:
    params.update({"num_groups": grp})
    for dtm in DETERMINISTIC:
        params.dqn.deterministic = dtm
        # run experiments
        env = MachineReplacement(
            num_groups=params.num_groups,
            num_states=params.num_states,
            num_actions=params.num_actions,
            num_steps=params.len_episode,
            encoding_int=params.dqn.encoding_int,
        )

        all_rewards = {}
        for ep in EXPLORATION_RATES:
            for lr in LEARNING_RATES:
                agent = DQNAgent(
                    env=env,
                    learning_rate=lr,
                    discount_factor=params.gamma,
                    exploration_rate=ep,
                    decaying_factor=params.dqn.decaying_factor,
                    deterministic=params.dqn.deterministic,
                )
                episode_rewards = agent.run(
                    num_episodes=params.num_episodes,
                    len_episode=params.len_episode,
                    num_samples=params.num_samples,
                )
                unique_key_str = f"{ep}-{lr}"
                all_rewards[unique_key_str] = episode_rewards

                if SAVE_RESULTS:
                    # file string
                    TIME_STAMP = str(datetime.now().strftime("%m%d%H%M%S"))
                    TYPE_INDICATOR = "D" if params.dqn.deterministic else "S"
                    RUN = (
                        f"A{agent.count_stat.is_deterministic_act}_I{agent.count_stat.is_deterministic_improve}_"
                        f"{params.num_episodes}_{params.len_episode}_{params.num_samples}"
                    )
                    # save the rewards
                    all_rewards_df = pd.DataFrame.from_dict(all_rewards)
                    all_rewards_df.to_csv(
                        f"results/rewards_dqn{TYPE_INDICATOR}_{params.num_groups}_{TIME_STAMP}_{RUN}.csv"
                    )
                    # save time statistics
                    with open(
                        f"results/time_stat_dqn{TYPE_INDICATOR}_{params.num_groups}_{TIME_STAMP}_{RUN}.json",
                        "w",
                    ) as fp:
                        json.dump(agent.time_stat, fp)
                    # save policy
                    policy_df = pd.DataFrame.from_dict(agent.policy, orient="index")
                    policy_df = policy_df.sort_index()
                    policy_df.to_csv(
                        f"results/policy_dqn{TYPE_INDICATOR}_{params.num_groups}_{TIME_STAMP}_{RUN}.csv"
                    )

if PLOT:
    # plot the rewards
    plot_figures(
        ep_list=EXPLORATION_RATES, lr_list=LEARNING_RATES, all_rewards=all_rewards
    )
