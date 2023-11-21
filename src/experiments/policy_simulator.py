"""This module contains the experiments to calculate and simulate the state value function."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params
from solver.ggf_dlp import build_dlp_fix, extract_dlp, solve_dlp
from utils.common import MDP4LP
from utils.mrp import MRPData
from utils.policy import calculate_state_value, simulation_state_value

# set up the parameters
run_simulation = False
params.num_samples = 100
# Notice: modify the file path to the one containing policy
policy = pd.read_csv("results/policy/deterministic_policy_group2.csv", index_col=0)

# size of the problem
state_size, action_size = policy.shape[0], policy.shape[1]

# truncate and normalize the policy
for state in range(state_size):
    policy.iloc[state] = policy.iloc[state].round(4)  # Truncate to 4 digits
    policy.iloc[state] /= policy.iloc[state].sum()  # Normalize

# set up the environment according to the policy
params.num_groups = action_size - 1
params.num_states = int(pow(state_size, 1 / params.num_groups))

# set up the environment
env = MachineReplacement(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    num_steps=params.len_episode,
    encoding_int=params.dqn.encoding_int,
)

# solve with given policy
mrp_data = MRPData(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    prob_remain=params.prob_remain,
    # weight_type="uniform",
    # cost_types_operation=["quadratic"] * params.num_groups,
    # cost_types_replace=["quadratic"] * params.num_groups,
)

mdp = MDP4LP(
    num_states=mrp_data.num_global_states,
    num_actions=mrp_data.num_global_actions,
    num_groups=mrp_data.num_groups,
    transition=mrp_data.global_transitions,
    costs=mrp_data.global_costs,
    discount=params.gamma,
    weights=mrp_data.weights,
    minimize=True,
    encoding_int=False,
    base_num_states=params.num_states,
)

# calculate LP values
model = build_dlp_fix(mdp=mdp, policy=policy)
# Solve the GGF model
_, model, _ = solve_dlp(model=model)
results = extract_dlp(model=model, print_results=True)

if run_simulation:
    # True if the state distribution set as uniform
    uniform_state_start = True
    if uniform_state_start:
        len_plot = 1
    else:
        # Define the number of rows and columns for subplots
        rows = int(np.ceil(np.sqrt(state_size)))
        cols = int(np.ceil(state_size / rows))
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
        len_plot = state_size

    # simulation
    params.len_episode = 100
    max_y = 0
    for state in range(len_plot):
        # set the initial state distribution
        if uniform_state_start:
            initial_state_prob = np.ones(state_size) / state_size
        else:
            initial_state_prob = np.zeros(state_size)
            initial_state_prob[state] = 1

        # calculate the state value
        state_value_list = calculate_state_value(
            discount=params.gamma,
            initial_state_prob=initial_state_prob,
            policy=policy.to_numpy(),
            reward_or_cost=mrp_data.global_costs,
            transition_prob=mrp_data.global_transitions,
            time_horizon=params.len_episode,
        )
        state_value_sim = simulation_state_value(
            params, policy, mrp_data, initial_state_prob
        )

        GGF_value_list, GGF_value_list_sim = [], []
        for t in range(params.len_episode):
            GGF_value_list.append(np.dot(mrp_data.weights, sorted(state_value_list[t])))
            GGF_value_list_sim.append(
                np.dot(mrp_data.weights, sorted(state_value_sim[t]))
            )

        # Plot GGF_value_list on the current subplot
        if uniform_state_start:
            sns.lineplot(data=GGF_value_list, label="Matrix")
            sns.lineplot(data=GGF_value_list_sim, label="Simulation")
            sns.lineplot(data=[results.ggf_value_ln] * params.len_episode, label="DLP")
        else:
            # Calculate subplot position
            row = state // cols
            col = state % cols

            ax = axes[row, col]
            ax.set_title(f"State {state}", size=8)  # Set subplot title
            sns.lineplot(data=GGF_value_list, ax=ax, label="Matrix")
            sns.lineplot(data=GGF_value_list_sim, ax=ax, label="Simulation")
            sns.lineplot(
                data=[results.ggf_value_ln] * params.len_episode, ax=ax, label="DLP"
            )
            ax.legend(fontsize=8)
        # Update parameter used to set the y-axis limit
        max_y = max(max_y, GGF_value_list[-1], GGF_value_list_sim[-1])

    # Show the subplots
    if uniform_state_start:
        plt.ylim([0, np.ceil(1.01 * max_y)])
        plt.xlim([0, params.len_episode])
        plt.xlabel("Time")
        plt.ylabel("Discounted GGF Value")
        plt.title(
            f"{params.num_groups} machines, {params.num_states} states, {params.num_actions} actions"
        )
    else:
        # Set the x- and y-axis limit for all subplots
        plt.setp(axes, xlim=[0, params.len_episode], ylim=[0, np.ceil(1.01 * max_y)])
        plt.suptitle(
            f"{params.num_groups} machines, {params.num_states} states, {params.num_actions} actions"
        )
    plt.show()
