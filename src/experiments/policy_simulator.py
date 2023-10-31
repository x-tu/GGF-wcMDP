import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params
from solver.dual_mdp import LPData, build_dlp_fix, extract_dlp, solve_dlp

# Notice: modify the file path to the one containing policy
policy = pd.read_csv("results/policy/policy_dqnS_3.csv", index_col=0)

# size of the problem
state_size = policy.shape[0]
action_size = policy.shape[1]

# set up the environment according to the policy
params.num_groups = action_size - 1
params.num_states = int(pow(state_size, 1 / params.num_groups))
params.prob_remain = np.linspace(start=0.5, stop=0.9, num=params.num_groups)
params.num_samples = 4000

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

model = build_dlp_fix(mrp_data, policy)
# Solve the GGF model
results, ggf_model = solve_dlp(model=model)
extract_dlp(model=ggf_model, lp_data=mrp_data)

# run LP
sample_rewards = []
for n in tqdm(range(params.num_samples)):
    init_state = random.randint(0, state_size - 1)
    state = env.reset(initial_state=init_state)
    total_reward = [0] * params.num_groups
    for t in range(params.len_episode):
        action_prob = policy.iloc[state].tolist()
        action = random.choices(range(len(action_prob)), weights=action_prob, k=1)[0]
        reward_lp = mrp_data.global_costs[state, action]
        next_observation = random.choices(
            range(state_size),
            weights=mrp_data.global_transitions[state, :, action],
            k=1,
        )[0]
        total_reward += params.gamma ** t * reward_lp
        state = next_observation
    sample_rewards.append(total_reward)
# get the expected rewards by averaging over samples, and then sort
rewards_sorted = np.sort(np.mean(sample_rewards, axis=0))
episode_rewards = np.dot(env.weights, rewards_sorted)
print(episode_rewards)
