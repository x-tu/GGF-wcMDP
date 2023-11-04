import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params
from solver.dual_mdp import LPData, build_dlp_fix, extract_dlp, solve_dlp
from utils.policy import calculate_state_value, simulation_state_value

# Notice: modify the file path to the one containing policy
policy = pd.read_csv("results/policy/policy_dqnS_3.csv", index_col=0)

# size of the problem
state_size = policy.shape[0]
action_size = policy.shape[1]

# truncate and normalize the policy
for state in range(state_size):
    policy.iloc[state] = policy.iloc[state].round(4)  # Truncate to 4 digits
    policy.iloc[state] /= policy.iloc[state].sum()  # Normalize

# set up the environment according to the policy
params.num_groups = action_size - 1
params.num_states = int(pow(state_size, 1 / params.num_groups))
params.prob_remain = np.linspace(start=0.5, stop=0.9, num=params.num_groups)
params.num_samples = 20

# set up the environment
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

# solve with given policy
mrp_data = LPData(
    num_arms=params.num_groups,
    num_states=params.num_states,
    rccc_wrt_max=params.rccc_wrt_max,
    prob_remain=params.prob_remain,
    mat_type=params.mat_type,
    weights=env.weights,
    discount=params.gamma,
    encoding_int=True,
)

# calculate LP values
model = build_dlp_fix(mrp_data, policy)
# Solve the GGF model
results, ggf_model = solve_dlp(model=model)
_, _, ggf_value = extract_dlp(model=ggf_model, lp_data=mrp_data)

# Define the number of rows and columns for subplots
rows = cols = int(np.ceil(np.sqrt(state_size)))
fig, axes = plt.subplots(rows, cols)

# simulation
params.len_episode = 100
max_y = 0
for state in range(state_size):
    # set the initial state distribution
    initial_state_prob = np.zeros(state_size)
    initial_state_prob[state] = 1

    # calculate the state value
    state_value, state_value_list = calculate_state_value(
        discount=params.gamma,
        initial_state_prob=initial_state_prob,
        policy=policy.to_numpy(),
        rewards=mrp_data.global_costs,
        transition_prob=mrp_data.global_transitions,
        time_horizon=params.len_episode,
    )
    costs_sorted, cost_sorted_list = simulation_state_value(
        params, policy, mrp_data, initial_state_prob
    )
    episode_rewards = np.dot(env.weights, costs_sorted)

    GGF_value_list, GGF_value_list_sim = [], []
    for t in range(params.len_episode):
        GGF_value_list.append(np.dot(env.weights, sorted(state_value_list[t])))
        GGF_value_list_sim.append(np.dot(env.weights, sorted(cost_sorted_list[t])))

    # Calculate subplot position
    row = state // cols
    col = state % cols

    # Plot GGF_value_list on the current subplot
    ax = axes[row, col]
    ax.set_title(f"State {state}")  # Set subplot title
    sns.lineplot(data=GGF_value_list, ax=ax, label="Matrix")
    sns.lineplot(data=GGF_value_list_sim, ax=ax, label="Simulation")
    sns.lineplot(data=[ggf_value] * params.len_episode, ax=ax, label="DLP")
    max_y = max(max_y, GGF_value_list[-1], GGF_value_list_sim[-1])

# Show the subplots
plt.setp(axes, xlim=[0, params.len_episode], ylim=[0, np.ceil(1.01 * max_y)])
plt.show()
