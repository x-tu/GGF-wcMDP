import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.network import ACNetwork


class A2CAgent:
    def __init__(
        self,
        env,
        policy_loss_weight,
        value_loss_weight,
        entropy_loss_weight,
        hidden_dims: tuple = (64, 64),
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        decaying_factor: float = 0.99,
        deterministic: bool = True,
        seed: int = 10,
    ):
        self.env = env
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.decaying_factor = decaying_factor
        self.weights = torch.tensor(self.env.weights, dtype=torch.float32)

        self.deterministic = deterministic

        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.num_groups = self.env.reward_space.n

        self.ac_network = ACNetwork(
            input_dim=self.num_groups,
            output_dim_policy=self.num_actions,
            output_dim_value=self.num_groups,
            hidden_dims=hidden_dims,
        )
        self.ac_optimizer = optim.RMSprop(
            self.ac_network.parameters(), lr=learning_rate
        )

        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        self.seed = seed

        self.log_probs = []
        self.entropies = []
        self.values = []
        self.running_exploration = 0

    def act(self, observation: np.array) -> int:
        """Select an action according to the observation.

        Args:
            observation (`np.array`): the current observable state.

        Returns:
            (`int`): the selected action.
        """

        action, is_exploratory, log_prob, entropy, values = self.ac_network.full_pass(
            state=observation
        )

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.values.append(values)
        self.running_exploration += is_exploratory[:, np.newaxis].astype(np.int)

        return action

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
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # reshape the next Q-values to a matrix of shape (action, group)
        values = self.values[-1]
        _, _, _, _, next_values = self.ac_network.full_pass(state=observation_next)

        # compute the value loss
        value_error = reward_tensor + self.gamma * next_values - values
        value_loss = value_error.pow(2).mul(0.5)

        # compute the policy loss
        policy_loss = 0

        # compute the entropy loss
        entropy_loss = -self.entropies[-1]

        loss = (
            self.policy_loss_weight * policy_loss
            + self.value_loss_weight * value_loss
            + self.entropy_loss_weight * entropy_loss
        )

        self.ac_optimizer.zero_grad()
        loss.backward()
        self.ac_optimizer.step()

    def evaluate(self):
        pass

    def run(
        self,
        num_episodes: int,
        len_episode: int,
        num_samples: int,
        initial_state_idx: int = None,
        random_seed: int = 10,
    ):

        """Train the agent for a number of episodes.

        Args:
            num_episodes (`int`): the number of episodes to train the agent.
            len_episode (`int`): the maximum length of an episode.
            num_samples (`int`): the number of samples to use for the GGI case.
            initial_state_idx (`int`): the initial state index to use.
            random_seed (`int`): the random seed to use for test consistency.
        """

        # record statistics
        episode_rewards = []

        # set the seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        for _ in tqdm(range(num_episodes)):
            ep_rewards = []
            for n in range(num_samples):
                initial_state = (
                    random.randint(0, self.env.observation_space.n - 1)
                    if not initial_state_idx
                    else initial_state_idx
                )
                observation = self.env.reset(
                    initial_state=initial_state, normalize=True
                )
                total_reward = np.zeros(self.env.reward_space.n)
                for t in range(len_episode):
                    action = self.act(observation=observation)
                    observation_next, reward, done, _ = self.env.step(action)
                    total_reward += (1 - done) * (self.gamma ** t) * reward
                    self.update(observation, action, reward, observation_next)
                    observation = observation_next
                ep_rewards.append(total_reward)
                # get the expected rewards by averaging over samples, and then sort
            rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
            episode_rewards.append(np.dot(self.weights, rewards_sorted))
            # # update the exploration rate
            # if self.exploration_rate > 0.001:
            #     self.exploration_rate = self.exploration_rate * self.decaying_factor
        # final_eval_score, score_std = self.evaluate()

        self.env.close()
        return episode_rewards  # , final_eval_score
