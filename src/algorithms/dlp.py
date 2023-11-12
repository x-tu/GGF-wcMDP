import random

import numpy as np
from pandas import DataFrame as df
from tqdm import tqdm

from env.mrp_env_rccc import MachineReplacement
from solver.ggf_dlp import build_dlp, extract_dlp, solve_dlp
from utils.common import MDP4LP, DotDict
from utils.encoding import state_vector_to_int_index


class DLPAgent:
    """MC simulation with Dual LP solutions for the MRP RCCC problem."""

    def __init__(self, params: DotDict):
        self.params = params
        self.env = MachineReplacement(
            num_groups=params.num_groups,
            num_states=params.num_states,
            num_actions=params.num_actions,
            num_steps=params.len_episode,
            prob_remain=params.prob_remain,
            encoding_int=params.dqn.encoding_int,
        )
        mdp = MDP4LP(
            num_states=self.env.mrp_data.num_global_states,
            num_actions=self.env.mrp_data.num_global_actions,
            num_groups=self.env.mrp_data.num_groups,
            transition=self.env.mrp_data.global_transitions,
            costs=self.env.mrp_data.global_costs,
            discount=self.params.gamma,
            weights=self.env.mrp_data.weights,
        )
        self.weights = self.env.mrp_data.weights

        model = build_dlp(mdp=mdp)
        results, model = solve_dlp(model)
        _, self.policy, self.ggf_value = extract_dlp(model, mdp)

        policy_df = df.from_dict(self.policy, orient="index")
        policy_df.to_csv(f"results/policy_dlp_{params.num_groups}.csv")

    def run_mc_dlp(self, initial_states: list = None):
        """Run the MC simulation.

        Notice: Random seeds may vary across different python versions or devices.
            We can force use a given initial state list if needed.

        Args:
            initial_states: list of initial states to run the simulation
        """

        episode_rewards = []
        for ep in tqdm(range(self.params.num_episodes)):
            # run LP
            ep_rewards = []
            for n in range(self.params.num_samples):
                init_state = random.randint(0, self.params.num_states - 1)
                # initial_state = initial_states[ep] if initial_states else 0
                state = self.env.reset(initial_state=init_state)
                total_reward = 0
                for t in range(self.params.len_episode):
                    state_idx = state_vector_to_int_index(
                        state_vector=state, num_states=self.params.num_states
                    )
                    action = np.random.choice(
                        range(len(self.policy[state_idx])), p=self.policy[state_idx]
                    )
                    next_observation, reward, done, _ = self.env.step(action)
                    total_reward += (1 - done) * self.params.gamma ** t * reward
                    state = (next_observation * self.env.num_states).astype(int)
                ep_rewards.append(total_reward)
            # get the expected rewards by averaging over samples, and then sort
            rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
            episode_rewards.append(np.dot(self.weights, rewards_sorted))
        return episode_rewards
