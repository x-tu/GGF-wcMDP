"""Implementation of the Multi-Objective Deep Q Network (DQN) Algorithm."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from solver.dual_q import get_policy_from_q_values


class DQNetwork(nn.Module):
    """Deep Q Network with 3 fully connected layers with ReLu activation function."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass."""
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Network Architecture: state -> Q-value[action, group]."""

    def __init__(
        self,
        env,
        h_size: int = 64,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        decaying_factor: float = 0.99,
        deterministic: bool = True,
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.decaying_factor = decaying_factor
        self.weights = env.weights
        self.deterministic = deterministic
        if not self.deterministic:
            # initialize the stochastic policy as uniform distribution
            self.policy = [1 / env.action_space.n] * env.action_space.n

        # input: states tuples that are encoded as integers
        input_dim = env.reward_space.n
        # output: Q-values
        output_dim = env.action_space.n * env.reward_space.n
        self.q_network = DQNetwork(
            input_dim=input_dim, hidden_dim=h_size, output_dim=output_dim
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, observation: np.array, reward_prev: np.array) -> int:
        """Select an action according to the observation.

        Args:
            observation (`np.array`): the current observable state.
            reward_prev (`np.array`): the previous reward.

        Returns:
            (`int`): the selected action.
        """
        # convert the reward vector to a column vector
        reward_prev_col = reward_prev.reshape((-1, 1))

        # 1) exploration
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        # 2) exploitation
        # reshape the Q-values to a matrix of shape (group |N|, action |A|)
        q_values = self.q_network(torch.tensor(observation).float()).reshape(
            (self.env.reward_space.n, self.env.action_space.n)
        )
        # deterministic policy is given by argmax GGF(r_prev + discount * q(s, a))
        if self.deterministic:
            # use vectorized computation to speed up the process
            q_values = q_values.detach().numpy()
            temp_q_values = reward_prev_col + self.discount_factor * q_values
            ggf_q_values = np.dot(self.env.weights, np.sort(temp_q_values, axis=0))
            return np.argmax(ggf_q_values).item()
        # stochastic policy is given by solving LP
        self.policy = get_policy_from_q_values(
            q_values=q_values.tolist(), weights=self.env.weights
        )
        return np.random.choice(range(self.env.action_space.n), p=self.policy)

    def update(
        self,
        observation: np.array,
        action: int,
        reward: np.array,
        observation_next: np.array,
    ):
        """Update the weights of the Q network.

        Args:
            observation (`np.array`): the current observable state.
            action (`int`): the selected action.
            reward (`np.array`): the reward.
            observation_next (`np.array`): the next observable state.
        """

        # Convert the weights and rewards to column PyTorch tensors
        weight_tensor = torch.tensor(self.env.weights, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # reshape the next Q-values to a matrix of shape (action, group)
        next_q_values = self.q_network(
            torch.tensor(observation_next, dtype=torch.float32)
        ).reshape((self.env.reward_space.n, self.env.action_space.n))
        q_values = self.q_network(
            torch.tensor(observation, dtype=torch.float32)
        ).reshape((self.env.reward_space.n, self.env.action_space.n))

        # get the current GGF values
        ggf_values = torch.matmul(weight_tensor, torch.sort(q_values, dim=0).values)
        target_ggf_values = ggf_values.clone()

        # update the weights of the Q network when the policy is deterministic
        if self.deterministic:
            # get the next best action with the highest GGI value (greedy)
            next_ggf_values = torch.matmul(
                weight_tensor, torch.sort(next_q_values, dim=0).values
            )
            action_best = torch.argmax(next_ggf_values).item()
            # update the target GGF values
            target_ggf_values[action] = torch.matmul(
                weight_tensor,
                torch.sort(
                    reward_tensor
                    + (self.discount_factor * next_q_values[:, action_best]).squeeze()
                ).values,
            )

        # update the weights of the Q network when the policy is stochastic
        else:
            # check first if we need to solve the LP
            policy_next = torch.tensor(
                get_policy_from_q_values(
                    q_values=next_q_values.tolist(), weights=self.env.weights
                ),
                dtype=torch.float32,
            )
            # update the target GGF values
            target_ggf_values[action] = torch.matmul(
                weight_tensor,
                torch.sort(
                    reward_tensor
                    + self.discount_factor * torch.matmul(next_q_values, policy_next)
                ).values,
            )
        # compute the loss
        loss = self.loss_fn(ggf_values, target_ggf_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, num_episodes: int, len_episode: int, num_samples: int):
        """Train the agent for a number of episodes.

        Args:
            num_episodes (`int`): the number of episodes to train the agent.
            len_episode (`int`): the maximum length of an episode.
            num_samples (`int`): the number of samples to use for the GGI case.
        """

        # record statistics
        episode_rewards = []
        for _ in tqdm(range(num_episodes)):
            ep_rewards = []
            for n in range(num_samples):
                observation = self.env.reset()
                reward = np.zeros(self.env.reward_space.n)
                total_reward = np.zeros(self.env.reward_space.n)
                for t in range(len_episode):
                    action = self.act(observation=observation, reward_prev=reward)
                    observation_next, reward, done, _ = self.env.step(action)
                    total_reward += (1 - done) * (self.discount_factor ** t) * reward
                    self.update(observation, action, reward, observation_next)
                    observation = observation_next
                ep_rewards.append(total_reward)
            # get the expected rewards by averaging over samples, and then sort
            rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
            episode_rewards.append(np.dot(self.env.weights, rewards_sorted))
            # update the exploration rate
            if self.exploration_rate > 0.001:
                self.exploration_rate = self.exploration_rate * self.decaying_factor
        return episode_rewards
