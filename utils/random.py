"""For count MDP. Run the random agent for a number of episodes."""

import numpy as np
from tqdm import tqdm


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def run(self, num_episodes=600, len_episode=300, runs=10):
        ep_rewards = np.zeros((num_episodes, self.env.num_groups))
        for ep in tqdm(range(num_episodes)):
            run_rewards = np.zeros((runs, self.env.num_groups))
            for run in range(runs):
                _ = self.env.reset()
                for t in range(len_episode):
                    action_priority = np.array(
                        [1 / self.env.count_mdp.num_states]
                        * self.env.count_mdp.num_states
                    )
                    if not self.env.force_to_use_all_resources:
                        action_priority += [np.random.uniform(0, 1)]
                    _, _, _, _ = self.env.step(action_priority)
                run_rewards[run, :] = self.env.last_group_rewards
            ep_rewards[ep, :] = run_rewards.mean(axis=0)
        return ep_rewards
