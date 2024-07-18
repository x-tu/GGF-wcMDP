import copy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from env.mrp_simulation import PropCountSimMDPEnv
from experiments.configs.base import params
from stable_baselines3 import PPO, SAC, TD3
from utils.common import update_params
from utils.count import CountMDP
from utils.policy import check_equal_means

# random agent
from utils.random import RandomAgent

ALGORITHMS = [PPO]
RUNS = 1000
RDM_NUM_EPISODES = 300
FILE_OUT = True
GROUPS = [10, 20, 50]
BUDGET_PROPS = [0.1, 0.2, 0.5]


def extract_policy(env_policy, model_policy):
    count_mdp = CountMDP(
        num_groups=params.num_groups,
        num_resource=params.budget,
        num_states=params.num_states,
    )
    # Print the policy
    for state in count_mdp.count_state_props:
        th_obs = torch.as_tensor(state).unsqueeze(0)
        env_policy.observations = state
        action_priority, _ = model_policy.predict(state, deterministic=True)
        count_action = env_policy.select_action_by_priority(action_priority)
        print(
            "state",
            state * params.num_groups,
            ": ",
            count_action,
            " | ",
            action_priority.round(2),
        )


def simulate_group_rewards(env_sim, model_sim, runs):
    rewards = np.zeros((runs, params.num_groups))
    for run in tqdm(range(runs)):
        state = env_sim.reset()
        for t in range(params.len_episode):
            # get the action
            action_priority, _ = model_sim.predict(state, deterministic=True)
            next_state, reward, done, info = env_sim.step(action_priority)
            state = next_state
        rewards[run, :] = env_sim.last_group_rewards
    print("Mean:", rewards.mean(axis=0).round(params.digit))
    print("Std:", rewards.std(axis=0).round(params.digit))
    return rewards


identifier = copy.deepcopy(params.identifier)
for group in GROUPS:
    params.num_groups = group
    for budget in BUDGET_PROPS:
        params.budget = int(budget * group)
        params = update_params(params, group, params.budget)

        print("Number of groups:", group, "Budget:", params.budget)
        # Evaluate the policy
        for algorithm in ALGORITHMS:
            env = PropCountSimMDPEnv(
                machine_range=[params.num_groups, params.num_groups],
                resource_range=[params.budget, params.budget],
                num_states=params.num_states,
                len_episode=params.len_episode,
                cost_types_operation=params.cost_type_operation,
                cost_types_replace=params.cost_type_replace,
                force_to_use_all_resources=params.force_to_use_all_resources,
            )
            identifier_to_load = (
                identifier if params.machine_range else params.identifier
            )
            print(f"Loading {identifier_to_load} model...")
            model = algorithm.load(
                f"experiments/tmp/{algorithm.__name__.lower()}_{identifier_to_load}"
            )
            # extract_policy(env, model)
            group_rewards = simulate_group_rewards(env, model, RUNS)
            rewards_df = pd.DataFrame(group_rewards)
            if FILE_OUT:
                rewards_df.to_csv(
                    f"experiments/tmp/rewards_{algorithm.__name__.lower()}_{params.identifier}.csv"
                )
            print(check_equal_means(groups=group_rewards.T))

            # calculate GGF
            print(
                f"GGF-{algorithm.__name__.lower()}: ",
                np.dot(
                    np.sort(rewards_df.mean().values), np.array(params.weights)
                ).round(params.digit),
            )

        env = PropCountSimMDPEnv(
            machine_range=[params.num_groups, params.num_groups],
            resource_range=[params.budget, params.budget],
            num_states=params.num_states,
            len_episode=params.len_episode,
            cost_types_operation=params.cost_type_operation,
            cost_types_replace=params.cost_type_replace,
            force_to_use_all_resources=params.force_to_use_all_resources,
        )
        random_agent = RandomAgent(env)
        rewards = random_agent.run(num_episodes=RDM_NUM_EPISODES)
        # calculate GGF
        print(
            f"GGF-random: ",
            np.dot(np.sort(rewards), params.weights).mean().round(params.digit),
        )
        if FILE_OUT:
            pd.DataFrame(rewards).to_csv(
                f"experiments/tmp/rewards_random_{params.identifier}.csv"
            )
