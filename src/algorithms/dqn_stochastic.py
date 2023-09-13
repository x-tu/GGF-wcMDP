import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.dqn_mrp import DQNetwork
from solver.dual_q import get_policy_from_q_values


class StochasticDQNAgent:
    """Network Architecture: (state, action) -> Q-value."""

    def __init__(
        self,
        env,
        discount,
        ggi_flag,
        weights,
        l_rate=1e-3,
        h_size=64,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.01,
    ):
        self.env = env
        self.ggi_flag = ggi_flag
        input_dim = env.num_arms
        output_dim = env.num_actions
        if ggi_flag:
            output_dim = env.num_arms * env.num_actions
        self.q_network = DQNetwork(input_dim, h_size, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=l_rate)
        self.weights = weights
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.loss_fn = nn.MSELoss()

        # used for statistics
        self.episode_rewards = []

    def predict(self, observation, prev_reward_list):
        """ Predict the Q-values for all actions for a given state.

        Returns:

        """
        # The Q-values
        q_values = self.q_network(torch.tensor(observation).float())
        if self.ggi_flag:
            # The Q-values for GGF case
            q_ggfvalues = torch.zeros(self.env.num_actions)
            # Loop over all actions
            for a_idx in range(self.env.num_actions):
                # The exploitation action is selected according to argmax_{a} GGF(r(s, a) + discount * q(s, a))
                temp_var = (
                    torch.tensor(prev_reward_list, dtype=torch.float32)
                    + self.discount
                    * q_values[
                        a_idx * self.env.num_arms : (a_idx + 1) * self.env.num_arms
                    ]
                )
                q_values_sorted = torch.sort(temp_var)
                for w in range(len(self.weights)):
                    q_ggfvalues[a_idx] += (
                        torch.tensor(self.weights)[w] * q_values_sorted[0][w]
                    )
            return torch.argmax(q_ggfvalues).item()
        else:
            return torch.argmax(q_values).item()

    def act(self, observation, prev_reward_list, deterministic=True):
        """Select an action according to the current policy.

        If the deterministic flag is set to True, then the action is selected according to the e-greedy policy based on
        Q values. Otherwise, the action is selected according to solving a dual LP.
        """

        if deterministic:
            # Exploration
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.env.num_actions)
            # Exploitation
            else:
                return self.predict(observation, prev_reward_list)
        else:
            # row: machine, column: action
            q_values = self.q_network(torch.tensor(observation).float()).reshape(
                (self.env.num_arms, self.env.num_actions)
            )
            policy = get_policy_from_q_values(
                q_values=q_values.tolist(), weights=self.env.weights
            )
            return random.choices(range(self.env.num_actions), weights=policy)[0]

    def update(self, observation, action, reward, next_observation):
        # The Q-values
        q_values = self.q_network(torch.tensor(observation).float())
        next_q_values = self.q_network(torch.tensor(next_observation).float())
        # For GGF case
        if self.ggi_flag:
            # Required variables for GGI case
            q_ggfvalues = torch.zeros(self.env.num_actions)
            next_q_ggfvalues = torch.zeros(self.env.num_actions)
            # Loop over all actions
            for a_idx in range(self.env.num_actions):
                temp_var = q_values[
                    a_idx * self.env.num_arms : (a_idx + 1) * self.env.num_arms
                ]
                next_temp_var = next_q_values[
                    a_idx * self.env.num_arms : (a_idx + 1) * self.env.num_arms
                ]
                q_ggfvalues_sorted = torch.sort(temp_var)
                nextq_ggfvalues_sorted = torch.sort(next_temp_var)
                for w in range(len(self.weights)):
                    q_ggfvalues[a_idx] += (
                        torch.tensor(self.weights)[w] * q_ggfvalues_sorted[0][w]
                    )
                    next_q_ggfvalues[a_idx] += (
                        torch.tensor(self.weights)[w] * nextq_ggfvalues_sorted[0][w]
                    )
            next_greedy_action = torch.argmax(next_q_ggfvalues).item()
            target_ggfsorted = torch.sort(
                torch.tensor(reward)
                + self.discount * next_q_ggfvalues[next_greedy_action]
            )
            target_ggf = 0
            for w in range(len(self.weights)):
                target_ggf += torch.tensor(self.weights)[w] * target_ggfsorted[0][w]
            target_q_ggfvalues = q_ggfvalues.clone()
            target_q_ggfvalues[action] = target_ggf
            # Compute the loss
            loss = self.loss_fn(q_ggfvalues, target_q_ggfvalues)
        else:
            # Compute the target: average reward over arms + discount * q(s', a_max) where
            # a_max = argmax_{a} q(s', a) is the greedy action for the next_state.
            next_greedy_action = torch.argmax(next_q_values).item()
            target = (
                np.dot(reward, self.weights)
                + self.discount * next_q_values[next_greedy_action].item()
            )
            target_q_values = q_values.clone()
            target_q_values[action] = target
            # Compute the loss
            loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # weight decaying
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def train(
        self,
        num_episodes: int,
        len_episode: int,
        num_samples: int,
        deterministic: bool = True,
    ):
        """Train the agent for a number of episodes.

        Args:
            num_episodes (`int`): the number of episodes to train the agent.
            len_episode (`int`): the maximum length of an episode.
            num_samples (`int`): the number of samples to use for the GGI case.
        """

        for _ in tqdm(range(num_episodes)):
            ep_rewards = []
            for sp in range(num_samples):
                observation = self.env.reset()
                prev_reward_list = [0] * self.env.reward_space.n
                crt_reward_list = [0] * self.env.reward_space.n
                for t in range(len_episode):
                    action = self.act(
                        observation, prev_reward_list, deterministic=deterministic
                    )
                    next_observation, reward_list, done, _ = self.env.step(action)
                    crt_reward_list += (self.discount ** t) * reward_list
                    if done:
                        break
                    else:
                        self.update(observation, action, reward_list, next_observation)
                        observation = next_observation
                        prev_reward_list = reward_list
                ep_rewards.append(crt_reward_list)
            # use numpy to calculate the mean is more computationally efficient
            sorted_rewards = np.sort(ep_rewards, axis=1)
            mean_rewards = np.mean(sorted_rewards, axis=0)
            self.episode_rewards.append(np.dot(mean_rewards, self.weights))
