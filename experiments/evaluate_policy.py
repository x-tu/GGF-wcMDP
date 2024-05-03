import torch
from tqdm import tqdm

from env.mrp_count import CountMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy

params.len_episode = 500

env = CountMDPEnv(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    len_episode=params.len_episode,
)

model = PPO.load("experiments/tmp/ppo")

import numpy as np

# ep_rewards = 0
# state = 0
# for t in tqdm(range(300)):
#     action, _ = model.predict(state, deterministic=True)
#     ac_idx = env.count_mdp.count_env_idx_mapping[(state, int(action))]
#     # get next state
#     next_state_prob = env.count_mdp.count_transitions[state, :, ac_idx]
#     next_state = np.random.choice(
#         np.arange(env.observation_space.n), p=next_state_prob
#     )
#     # get the reward
#     reward = env.count_mdp.count_rewards[state, ac_idx]
#     ep_rewards += (params.gamma ** t) * reward
#     state = next_state
# print(-ep_rewards / params.num_groups)

# Print the policy
for state in range(len(env.count_mdp.count_states)):
    th_obs = torch.as_tensor(state).unsqueeze(0)
    try:
        probs = (
            model.policy.get_distribution(th_obs)
            .distribution.probs[0]
            .detach()
            .numpy()
            .round(4)
        )
    except:
        action_idx = model.policy.q_net._predict(th_obs)
        probs = np.zeros(env.action_space.n)
        probs[action_idx] = 1
    print("state", state, ": ", probs)
