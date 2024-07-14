import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from env.mrp_simulation import PropCountSimMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from utils.count import CountMDP
from utils.policy import check_equal_means

algorithms = [PPO, SAC, TD3]
runs = 10
file_out = False


def extract_policy(env, model):
    count_mdp = CountMDP(
        num_groups=params.num_groups,
        num_resource=params.budget,
        num_states=params.num_states,
    )
    # Print the policy
    for state in count_mdp.count_state_props:
        th_obs = torch.as_tensor(state).unsqueeze(0)
        env.observations = state
        action_priority, _ = model.predict(state, deterministic=True)
        count_action = env.select_action_by_priority(action_priority)
        print(
            "state",
            state * params.num_groups,
            ": ",
            count_action,
            " | ",
            action_priority.round(2),
        )


def simulate_group_rewards(env, model, runs):
    group_rewards = np.zeros((runs, params.num_groups))
    for run in tqdm(range(runs)):
        state = env.reset()
        for t in range(params.len_episode):
            # get the action
            action_priority, _ = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action_priority)
            state = next_state
        group_rewards[run, :] = env.last_group_rewards
    print("Mean:", group_rewards.mean(axis=0).round(params.digit))
    print("Std:", group_rewards.std(axis=0).round(params.digit))
    return group_rewards


# Evaluate the policy
for algorithm in algorithms:
    env = PropCountSimMDPEnv(
        machine_range=[params.num_groups, params.num_groups],
        resource_range=[params.budget, params.budget],
        num_states=params.num_states,
        len_episode=params.len_episode,
        cost_types_operation=params.cost_type_operation,
        cost_types_replace=params.cost_type_replace,
        force_to_use_all_resources=params.force_to_use_all_resources,
    )
    model = algorithm.load(
        f"experiments/tmp/{algorithm.__name__.lower()}_{params.identifier}.zip"
    )
    # extract_policy(env, model)
    group_rewards = simulate_group_rewards(env, model, runs=runs)
    rewards_df = pd.DataFrame(group_rewards)
    if file_out:
        rewards_df.to_csv(
            f"experiments/tmp/rewards_{algorithm.__name__.lower()}_{params.identifier}.csv"
        )
    print(check_equal_means(groups=group_rewards.T))

    weights = np.array([1 / (2**i) for i in range(params.num_groups)])
    weights /= np.sum(weights)
    # calculate GGF
    print(
        f"GGF-{algorithm.__name__.lower()}: ",
        np.dot(np.sort(rewards_df.mean().values), np.array(weights)).round(
            params.digit
        ),
    )
