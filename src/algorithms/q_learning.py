"""Implementation of the Multi-objective Q-learning Algorithm."""

from datetime import datetime

import numpy as np
from tqdm import tqdm

from solver.dual_q import get_policy_from_q_values, test_deterministic_optimal
from utils.common import DotDict


class QAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        decaying_factor: float,
        optimistic_start: [int, int] = None,
        deterministic: bool = True,
    ):
        self.env = env
        # initialize the q table
        if optimistic_start is None:
            optimistic_start = [0, 1]
        self.q_table = np.random.uniform(
            low=optimistic_start[0],
            high=optimistic_start[1],
            size=(env.observation_space.n, env.reward_space.n, env.action_space.n),
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.decaying_factor = decaying_factor
        self.lr_decay_schedule = []
        self.deterministic = deterministic
        # statistics for counting
        self.count_stat = DotDict(
            {"is_deterministic_act": 0, "is_deterministic_improve": 0}
        )
        # statistics for timing
        self.time_stat = DotDict(
            {"total": 0, "check_deterministic": [], "solve_LP": []}
        )

    def act(self, observation: int, reward_prev: np.array) -> int:
        """Get an action from the Q table with e-greedy strategy.

        Args:
            observation (`int`): the current observation of the environment.
            reward_prev (`np.array`): the reward list from the previous steps.

        Returns:
            action (`int`): the action to take.
        """

        # exploration
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        # exploitation (use vectorized computation to speed up)
        if self.deterministic:
            temp_q_values = (
                reward_prev.reshape((-1, 1))
                + self.discount_factor * self.q_table[observation, :, :]
            )
            ggf_q_values = np.dot(self.env.weights, np.sort(temp_q_values, axis=0))
            return np.argmax(ggf_q_values).item()
        # stochastic policy is given by solving LP, check first if we need to solve the LP
        is_deterministic_optimal, a_idx = test_deterministic_optimal(
            q_values=self.q_table[observation, :, :], weights=self.env.weights
        )
        if is_deterministic_optimal:
            self.count_stat.is_deterministic_act += 1
            return a_idx
        # solve LP
        policy = get_policy_from_q_values(
            q_values=self.q_table[observation, :, :], weights=self.env.weights
        )
        return np.random.choice(range(self.env.action_space.n), p=policy)

    def update(
        self, observation: int, action: int, reward: np.array, observation_next: int
    ):
        """Update the Q table.

        Args:
            observation (`int`): the current observation of the environment.
            action (`int`): the action to take.
            reward (`np.array`): the reward for taking the action.
            observation_next (`int`): the next observation of the environment.
        """

        # get the next action with the highest GGI value (greedy)
        action_best = np.argmax(
            np.dot(
                self.env.weights, np.sort(self.q_table[observation_next, :, :], axis=0)
            )
        ).item()
        if self.deterministic:
            # update the Q table
            self.q_table[observation, :, action] += self.learning_rate * (
                reward
                + self.discount_factor * self.q_table[observation_next, :, action_best]
                - self.q_table[observation, :, action]
            )
        # update the values of the Q table when the policy is stochastic
        else:
            # check first if we need to solve the LP
            is_deterministic_optimal, a_idx = test_deterministic_optimal(
                q_values=self.q_table[observation, :, :], weights=self.env.weights
            )
            if is_deterministic_optimal:
                self.count_stat.is_deterministic_improve += 1
                policy_next = np.zeros(self.env.action_space.n)
                policy_next[a_idx] = 1
            else:
                start_time = datetime.now()
                policy_next = get_policy_from_q_values(
                    q_values=self.q_table[observation_next, :, :],
                    weights=self.env.weights,
                )
                self.time_stat.solve_LP.append(
                    (datetime.now() - start_time).total_seconds()
                )
            # update the Q table
            self.q_table[observation, :, action] += self.learning_rate * (
                reward
                + self.discount_factor
                * np.dot(self.q_table[observation_next, :, :], policy_next)
                - self.q_table[observation, :, action]
            )

    def run(self, num_episodes: int, len_episode: int, num_samples: int):
        # set the linear decaying schedule for learning rate
        self.lr_decay_schedule = np.linspace(
            start=self.learning_rate, stop=0, num=num_episodes
        )

        # record statistics
        episode_rewards = []
        for ep in tqdm(range(num_episodes)):
            ep_rewards = []
            for n_idx in range(num_samples):
                observation = self.env.reset()
                reward = np.zeros(self.env.reward_space.n)
                total_reward = np.zeros(self.env.reward_space.n)
                for t_idx in range(len_episode):
                    action = self.act(observation=observation, reward_prev=reward)
                    observation_next, reward, done, _ = self.env.step(action)
                    total_reward += (1 - done) * self.discount_factor ** t_idx * reward
                    self.update(observation, action, reward, observation_next)
                    observation = observation_next
                ep_rewards.append(total_reward)
            # get the expected rewards by averaging over samples, and then sort
            rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
            episode_rewards.append(np.dot(self.env.weights, rewards_sorted))
            # update the learning rate
            self.learning_rate = self.lr_decay_schedule[ep]
            # update the exploration rate
            if self.exploration_rate > 0.001:
                self.exploration_rate = self.exploration_rate * self.decaying_factor
        return episode_rewards
