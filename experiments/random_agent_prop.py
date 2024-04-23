import pandas as pd
import torch
from tqdm import tqdm

from env.mrp_count import CountMDPEnv
from env.mrp_prop_count import PropCountMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, DDPG, DQN, PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy

params.len_episode = 500

env = PropCountMDPEnv(
    machine_range=[2, 2],
    resource_range=[1, 1],
    num_states=params.num_states,
    len_episode=params.len_episode,
)

import numpy as np

# the count mdp model to be evaluated
# count_mdp = env.count_mdp_pool[5]
# state = count_mdp.count_state_props[0]

random_rewards = []
for ep in tqdm(range(600)):
    ep_rewards = 0
    state = env.reset()
    for t in range(300):
        # get the action
        action = [1 / env.count_mdp.num_states] * env.count_mdp.num_states + [
            np.random.uniform(0, 1)
        ]
        count_action = env.select_action_by_priority(np.array(action))
        action_idx = env.count_mdp.ac_to_idx_mapping[str(count_action)]
        # get state
        count_state = state[: env.count_mdp.num_states] * env.count_mdp.num_groups
        state_idx = env.count_mdp.mapping_x_to_idx[str(list(count_state.astype(int)))]
        # transit to the next state
        next_state_prob = env.count_mdp.count_transitions[state_idx, :, action_idx]
        next_state_idx = np.random.choice(
            np.arange(len(next_state_prob)), p=next_state_prob
        )
        next_state = env.count_mdp.count_state_props[next_state_idx]
        # get the reward
        reward = (
            env.count_mdp.count_rewards[state_idx, action_idx]
            / env.count_mdp.num_groups
            + env.reward_offset
        )
        ep_rewards += (params.gamma**t) * reward
        state = next_state
        env.observations = next_state
    random_rewards.append(ep_rewards)
df = pd.DataFrame(np.array(random_rewards))
df.to_csv("experiments/tmp/random.csv")


# Print the policy
# for state in count_mdp.count_state_props:
#     th_obs = torch.as_tensor(state).unsqueeze(0)
#     try:
#         probs = model.policy.get_distribution(th_obs).distribution.probs[0].detach().numpy().round(4)
#     except:
#         action_idx = model.policy.q_net._predict(th_obs)
#         probs = np.zeros(env.action_space.n)
#         probs[action_idx] = 1
#     print("state", state, ": ", probs)
