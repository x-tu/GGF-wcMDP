import numpy as np
from tqdm import tqdm


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.ep_rewards = []

    def run(self, num_episodes=600, len_episode=300):
        for _ in tqdm(range(num_episodes)):
            ep_reward = 0
            state = self.env.reset()
            for t in range(len_episode):
                action = [
                    1 / self.env.count_mdp.num_states
                ] * self.env.count_mdp.num_states + [np.random.uniform(0, 1)]
                count_action = self.env.select_action_by_priority(np.array(action))
                count_state = (
                    state[: self.env.num_states] * self.env.num_groups
                ).astype(int)

                # simulate the next state
                next_count_state = np.zeros_like(count_state)
                reward = 0
                for i in range(self.env.num_states):
                    for j in range(count_state[i]):
                        action_idx = 1 if count_action[i] > 0 else 0
                        count_action[i] -= 1 if count_action[i] > 0 else 0
                        next_state_index = np.random.choice(
                            range(self.env.num_states),
                            p=self.env.count_mdp.global_transitions[i, :, action_idx],
                        )
                        next_count_state[next_state_index] += 1
                        reward += (
                            self.env.count_mdp.global_rewards[i, action_idx, 0]
                            + self.env.reward_offset
                        )
                reward /= self.env.num_groups
                ep_reward += (self.env.gamma**t) * reward
                state = (
                    np.concatenate((next_count_state, [self.env.num_budget]))
                    / self.env.num_groups
                )
            self.ep_rewards.append(ep_reward)
