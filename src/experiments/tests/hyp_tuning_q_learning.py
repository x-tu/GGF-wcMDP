import json
from datetime import datetime

import pandas as pd

from algorithms.q_learning import QAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params

# for batching the experiments
for grp in [3, 4]:
    params.update({"num_groups": grp})
    for dtm in [False]:
        params.dqn.deterministic = dtm
        env = MachineReplacement(
            num_groups=params.num_groups,
            num_states=params.num_states,
            num_actions=params.num_actions,
            num_steps=params.len_episode,
            encoding_int=params.dqn.encoding_int,
        )

        learning_rates = [0.3]
        exploration_rates = [0.7]

        all_rewards = {}
        for ep in exploration_rates:
            for lr in learning_rates:
                agent = QAgent(
                    env=env,
                    learning_rate=lr,
                    discount_factor=params.gamma,
                    exploration_rate=ep,
                    decaying_factor=params.ql.decaying_factor,
                    deterministic=params.ql.deterministic,
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
        TYPE_INDICATOR = "D" if params.ql.deterministic else "S"
        RUN = (
            f"A{agent.count_stat.is_deterministic_act}_I{agent.count_stat.is_deterministic_improve}_"
            f"{params.num_episodes}_{params.len_episode}_{params.num_samples}"
        )
        # save the rewards
        all_rewards_df = pd.DataFrame.from_dict(all_rewards)
        all_rewards_df.to_csv(
            f"results/rewards_ql{TYPE_INDICATOR}_{params.num_groups}_{TIME_STAMP}_{RUN}.csv"
        )
        # save time statistics
        with open(
            f"results/time_stat_ql{TYPE_INDICATOR}_{params.num_groups}_{TIME_STAMP}_{RUN}.json",
            "w",
        ) as fp:
            json.dump(agent.time_stat, fp)
        # save policy
        policy_df = pd.DataFrame.from_dict(agent.policy, orient="index")
        policy_df = policy_df.sort_index()
        policy_df.to_csv(
            f"results/policy_ql{TYPE_INDICATOR}_{params.num_groups}_{TIME_STAMP}_{RUN}.csv"
        )
