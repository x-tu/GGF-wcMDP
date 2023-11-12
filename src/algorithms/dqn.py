"""Implementation of the Multi-Objective Deep Q Network (DQN) Algorithm."""

import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyomo.opt import SolverFactory
from tqdm import tqdm

from solver.dual_q import build_dual_q_model, test_deterministic_optimal
from utils.common import DotDict
from utils.encoding import state_vector_to_int_index


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
        self.weights = self.env.mrp_data.weights
        self.deterministic = deterministic
        if not self.deterministic:
            # initialize the stochastic policy as uniform distribution
            state_policy = [1 / env.action_space.n] * env.action_space.n
        else:
            # do nothing by default
            state_policy = [1] + [0] * (env.action_space.n - 1)
        self.policy = {s: state_policy for s in range(env.observation_space.n)}

        # input: states tuples that are encoded as integers
        input_dim = env.reward_space.n
        # output: Q-values
        output_dim = env.action_space.n * env.reward_space.n
        self.q_network = DQNetwork(
            input_dim=input_dim, hidden_dim=h_size, output_dim=output_dim
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        # statistics for counting
        self.count_stat = DotDict(
            {"is_deterministic_act": 0, "is_deterministic_improve": 0}
        )
        # statistics for timing
        self.time_stat = DotDict(
            {
                "total": 0,
                "episode": [],
                "sample": [],
                "step": [],
                "act": [],
                "check_dtm_act": [],
                "solve_lp_act": [],
                "env": [],
                "improve": [],
                "check_dtm_improve": [],
                "solve_lp_improve": [],
            }
        )
        # initialize the model and solver
        self.model = None
        self.solver = None

    def act(self, observation: np.array) -> int:
        """Select an action according to the observation.

        Args:
            observation (`np.array`): the current observable state.

        Returns:
            (`int`): the selected action.
        """

        # 1) exploration
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        # 2) exploitation
        # reshape the Q-values to a matrix of shape (group |N|, action |A|)
        q_values = (
            self.q_network(torch.tensor(observation).float())
            .reshape((self.env.reward_space.n, self.env.action_space.n))
            .detach()
            .numpy()
        )
        # deterministic policy is given by argmax GGF(r_prev + discount * q(s, a))
        if self.deterministic:
            # use vectorized computation to speed up the process
            ggf_q_values = np.dot(self.weights, np.sort(q_values, axis=0))
            return np.argmax(ggf_q_values).item()
        start_time = datetime.now()
        # stochastic policy is given by solving LP, check first if we need to solve the LP
        is_deterministic_optimal, a_idx = test_deterministic_optimal(
            q_values=q_values, weights=self.weights
        )
        self.time_stat.check_dtm_act.append(
            (datetime.now() - start_time).total_seconds()
        )
        if is_deterministic_optimal:
            self.count_stat.is_deterministic_act += 1
            return a_idx
        start_time = datetime.now()

        # update the model parameters, notice that we are accessing the private attribute
        self.model.qvalues._data.update(
            {
                (d, a): q_values[d][a]
                for d in range(self.env.num_groups)
                for a in range(self.env.num_groups + 1)
            }
        )
        # TODO: remove this assertion after code being fully tested
        assert (
            self.model.qvalues._data[0, 0] == q_values[0, 0]
        ), f"Q values not updated correctly. {self.model.qvalues._data[0, 0].value} vs. {q_values[0, 0]}"
        self.solver.solve(self.model, tee=False)
        policy_next = [self.model.varP[a].value for a in self.model.varP]

        self.time_stat.solve_lp_act.append(
            (datetime.now() - start_time).total_seconds()
        )
        try:
            action = np.random.choice(range(self.env.action_space.n), p=policy_next)
        except ValueError:
            # for manual checking
            print("Negative probabilities in policy_next: ", policy_next)
            print("q_values: ", q_values)
            # temp solution to remove negative probabilities (caused by floating point errors in Pyomo)
            policy_next = [p if p > 0 else 0 for p in policy_next]
            # normalize the policy probabilities
            policy_next = [p / sum(policy_next) for p in policy_next]
            action = np.random.choice(range(self.env.action_space.n), p=policy_next)
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
        weight_tensor = torch.tensor(self.weights, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # reshape the next Q-values to a matrix of shape (action, group)
        next_q_values = self.q_network(
            torch.tensor(observation_next, dtype=torch.float32)
        ).reshape((self.env.reward_space.n, self.env.action_space.n))
        q_values = self.q_network(
            torch.tensor(observation, dtype=torch.float32)
        ).reshape((self.env.reward_space.n, self.env.action_space.n))

        # update the weights of the Q network when the policy is deterministic
        if self.deterministic:
            # get the next best action with the highest GGI value (greedy)
            next_ggf_values = torch.matmul(
                weight_tensor, torch.sort(next_q_values, dim=0).values
            )
            action_best = torch.argmax(next_ggf_values).item()
            target_q_values = (
                reward_tensor
                + (self.discount_factor * next_q_values[:, action_best]).squeeze()
            )
            policy_next = np.zeros(self.env.action_space.n)
            policy_next[action_best] = 1
        # update the weights of the Q network when the policy is stochastic
        else:
            start_time = datetime.now()
            # check first if we need to solve the LP
            is_deterministic_optimal, a_idx = test_deterministic_optimal(
                q_values=next_q_values.detach().numpy(), weights=self.weights
            )
            self.time_stat.check_dtm_improve.append(
                (datetime.now() - start_time).total_seconds()
            )
            if is_deterministic_optimal:
                self.count_stat.is_deterministic_improve += 1
                policy_next = torch.zeros(self.env.action_space.n)
                policy_next[a_idx] = 1
            else:
                start_time = datetime.now()
                # solve the updated model
                q_values_np = q_values.detach().numpy()
                self.model.qvalues._data.update(
                    {
                        (d, a): q_values_np[d][a]
                        for d in range(self.env.num_groups)
                        for a in range(self.env.num_groups + 1)
                    }
                )
                self.solver.solve(self.model, tee=False)
                policy_next = torch.tensor(
                    [self.model.varP[a].value for a in self.model.varP],
                    dtype=torch.float32,
                )
                self.time_stat.solve_lp_improve.append(
                    (datetime.now() - start_time).total_seconds()
                )
            # update the target GGF values
            target_q_values = reward_tensor + self.discount_factor * torch.matmul(
                next_q_values, policy_next
            )
        # convert the observation vector encoding to state index
        state = state_vector_to_int_index(
            state_vector=observation * self.env.num_states,
            num_states=self.env.num_states,
        )
        self.policy[state] = policy_next.tolist()
        # compute the loss
        loss = self.loss_fn(q_values[:, action], target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        start_time = datetime.now()
        episode_rewards = []
        states = []
        random.seed(random_seed)

        # build the LP model only once
        q_values = np.zeros((self.env.reward_space.n, self.env.action_space.n))
        self.model = build_dual_q_model(q_values=q_values, weights=self.weights)
        self.solver = SolverFactory("gurobi", solver_io="python")

        for _ in tqdm(range(num_episodes)):
            inner_start_time = datetime.now()
            ep_rewards = []
            for n in range(num_samples):
                initial_state = (
                    random.randint(0, self.env.observation_space.n - 1)
                    if not initial_state_idx
                    else initial_state_idx
                )
                # record for sanity check
                states.append(initial_state)
                sample_start_time = datetime.now()
                observation = self.env.reset(
                    initial_state=initial_state, normalize=True
                )
                total_reward = np.zeros(self.env.reward_space.n)
                for t in range(len_episode):
                    step_start_time = datetime.now()
                    action = self.act(observation=observation)
                    env_start_time = datetime.now()
                    observation_next, reward, done, _ = self.env.step(action)
                    total_reward += (1 - done) * (self.discount_factor ** t) * reward
                    update_start_time = datetime.now()
                    self.update(observation, action, reward, observation_next)
                    observation = observation_next
                    end_time = datetime.now()
                    self.time_stat.act.append(
                        (env_start_time - step_start_time).total_seconds()
                    )
                    self.time_stat.env.append(
                        (update_start_time - env_start_time).total_seconds()
                    )
                    self.time_stat.improve.append(
                        (end_time - update_start_time).total_seconds()
                    )
                    self.time_stat.step.append(
                        (end_time - step_start_time).total_seconds()
                    )
                ep_rewards.append(total_reward)
                self.time_stat.sample.append(
                    (datetime.now() - sample_start_time).total_seconds()
                )
            # get the expected rewards by averaging over samples, and then sort
            rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
            episode_rewards.append(np.dot(self.weights, rewards_sorted))
            # update the exploration rate
            if self.exploration_rate > 0.001:
                self.exploration_rate = self.exploration_rate * self.decaying_factor
            self.time_stat.episode.append(
                (datetime.now() - inner_start_time).total_seconds()
            )
        self.time_stat.total = (datetime.now() - start_time).total_seconds()
        print("\n", states)
        return episode_rewards
