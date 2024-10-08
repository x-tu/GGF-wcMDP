import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from algorithms.whittle import Whittle
from experiments.configs.base import params
from utils.common import update_params
from utils.count import CountMDP, count_to_normal
from utils.mrp import MRPData
from utils.policy import check_equal_means

N_RUNS = 1000
FILE_OUTPUT = False
PLOT_BAR = False
GROUPS = [10, 50, 100]
BUDGET_PROPS = [0.1, 0.2, 0.5]


def build_whittle_agent_with_gcd():
    # Find the GCD of the number of groups and the budget
    gcd = math.gcd(params.num_groups, params.budget)
    reduced_num_groups = params.num_groups // gcd
    reduced_budget = params.budget // gcd

    print(f"Reduced num_groups: {reduced_num_groups}, Reduced budget: {reduced_budget}")

    mrp = MRPData(
        num_groups=1,
        num_states=params.num_states,
        cost_types_operation=params.cost_type_operation,
        cost_types_replace=params.cost_type_replace,
    )
    # repeat the rewards and transitions for the number of groups (Only for homogeneous sub-MDPs. Check before use.)
    rewards = np.tile(mrp.rewards + 1, (params.num_groups, 1, 1))
    transitions = np.tile(mrp.transitions, (params.num_groups, 1, 1, 1))

    agent = Whittle(
        num_states=params.num_states,
        num_arms=reduced_num_groups,
        reward=rewards,  # (arm, x, act)
        transition=transitions,  # (arm, x, x', act)
        horizon=params.len_episode,
    )

    # try to load the whittle index if it exists
    try:
        whittle_identifier = (
            f"G{reduced_num_groups}_"
            f"C{params.cost_type_operation[:2]}-{params.cost_type_replace[:2]}_"
            f"F{'o' if params.ggi else 'x'}_"
            f"K{reduced_budget}{'o' if params.force_to_use_all_resources else 'x'}"
        )
        # the header and index is not needed
        whittle_indices = pd.read_csv(
            f"experiments/tmp/whittle_index_{whittle_identifier}.csv", index_col=0
        ).values
    except FileNotFoundError:
        agent.whittle_brute_force(lower_bound=-2, upper_bound=2, num_trials=1000)
        whittle_indices = agent.w_indices[0]
        # save the whittle index
        pd.DataFrame(agent.w_indices[0]).to_csv(
            f"experiments/tmp/whittle_index_{params.identifier}.csv"
        )

    # Update the indices to the original number of groups
    agent.num_a = params.num_groups
    agent.w_indices = [whittle_indices for _ in range(params.num_groups)]
    return agent, mrp


def run_whittle_mc_evaluation(agent, mrp):
    # MC simulation to evaluate the policy
    group_rewards = np.zeros((params.num_groups, N_RUNS))
    for run in tqdm(range(N_RUNS)):
        state = np.random.choice(range(params.num_states), size=params.num_groups)
        for t in range(params.len_episode):
            action = agent.Whittle_policy(
                whittle_indices=agent.w_indices,
                n_selection=params.budget,
                current_x=state,
                current_t=t,
                shuffle_indices=params.ggi,
                force_to_use_all_resources=params.force_to_use_all_resources,
            )
            # get the reward
            next_state = np.zeros_like(state)
            for arm in range(params.num_groups):
                group_rewards[arm, run] += (
                    mrp.rewards[0, state[arm], action[arm]] + 1
                ) * (params.gamma**t)
                next_state[arm] = np.random.choice(
                    range(params.num_states),
                    p=mrp.transitions[0, state[arm], :, action[arm]],
                )
            # print(state, action, group_rewards[:, run], next_state)
            state = next_state

    rewards = pd.DataFrame(group_rewards.T)
    rewards.columns = [f"Machine {i + 1}" for i in range(params.num_groups)]
    if PLOT_BAR:
        # plot the mean and variance of the two groups
        rewards.mean().plot(kind="bar", yerr=rewards.std())
        # label is horizontal
        plt.xticks(rotation=0)
        plt.show()
    print("Mean:", rewards.mean().values.round(params.digit))
    print("Std:", rewards.std().values.round(params.digit))
    # calculate GGF
    print(
        "GGF: ",
        np.dot(np.sort(rewards.mean().values), np.array(params.weights)).round(4),
    )
    print(check_equal_means(group_rewards))
    return rewards


def get_policy(agent):

    count_mdp = CountMDP(
        num_groups=params.num_groups,
        num_states=params.num_states,
        num_resource=params.budget,
    )

    policy = {}
    # Print the policy
    for state in count_mdp.count_state_props:
        count_state = (state * agent.num_a)[: agent.num_x]
        # convert count state to state
        normal_state = count_to_normal(count_state)
        normal_action = agent.Whittle_policy(
            whittle_indices=agent.w_indices,
            n_selection=params.budget,
            current_x=normal_state,
            current_t=0,
            shuffle_indices=params.ggi,
            force_to_use_all_resources=params.force_to_use_all_resources,
        )
        # count action
        count_action = count_mdp.action_normal_to_count(normal_action, normal_state)
        policy[str(count_state)] = count_action
        print(count_state, count_action)
    return policy


for group in GROUPS:
    params.num_groups = group
    for budget_prop in BUDGET_PROPS:
        params.budget = int(group * budget_prop)
        params = update_params(params, group, params.budget)

        print(f"Number of groups: {params.num_groups}, Budget: {params.budget}")

        whittle_agent, mrp_data = build_whittle_agent_with_gcd()
        rewards_df = run_whittle_mc_evaluation(whittle_agent, mrp_data)

        if FILE_OUTPUT:
            # save the rewards
            rewards_df.to_csv(
                f"experiments/tmp/rewards_whittle_{params.identifier}.csv", index=False
            )
            # transpose the dictionary to save the policy
            pd.DataFrame.from_dict(get_policy(whittle_agent), orient="index").to_csv(
                f"experiments/tmp/policy_whittle_{params.identifier}.csv"
            )
