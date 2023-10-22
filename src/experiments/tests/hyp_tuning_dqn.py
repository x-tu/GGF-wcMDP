"""This script is used to test the hyperparameter tuning for DQN."""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from algorithms.dqn import DQNAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# params.update({"num_episodes": 20, "len_episode": 10, "num_samples": 1})

# for batching the experiments
for grp in [6]:
    params.update({"num_groups": grp})
    params.update(
        {"prob_remain": np.linspace(start=0.5, stop=0.9, num=params.num_groups)}
    )
    for dtm in [False]:
        params.dqn.deterministic = dtm
        # run experiments
        env = MachineReplacement(
            num_arms=params.num_groups,
            num_states=params.num_states,
            rccc_wrt_max=params.rccc_wrt_max,
            prob_remain=params.prob_remain,
            mat_type=params.mat_type,
            weight_coefficient=params.weight_coefficient,
            num_steps=params.len_episode,
            ggi=params.ggi,
            encoding_int=params.dqn.encoding_int,
        )

        learning_rates = [1e-4]
        exploration_rates = [0.6]

        all_rewards = {}
        for ep in exploration_rates:
            for lr in learning_rates:
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
