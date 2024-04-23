# temporarily disable the warnings
import warnings
from typing import Union

import gym
import numpy as np
from gym import spaces

from utils.count import CountMDP, softmax

warnings.filterwarnings("ignore")


class PropCountSimMDPEnv(gym.Env):
    """A gym environment for the machine replacement problem."""

    def __init__(
        self,
        num_states: int,
        machine_range=None,
        resource_range=None,
        len_episode: int = 300,
        gamma: float = 0.95,
        rccc_wrt_max: float = 1.5,
        prob_remain: float = 0.8,
        deterioration_step: int = 1,
    ):
        super(gym.Env, self).__init__()

        # assign parameters
        self.observations = None
        self.num_groups = None
        self.num_budget = None
        if machine_range is None:
            machine_range = [2, 5]
        self.range_d = list(range(machine_range[0], machine_range[1] + 1))
        if resource_range is None:
            resource_range = [1, 1]
        self.range_k = list(range(resource_range[0], resource_range[1] + 1))

        self.num_states = num_states
        self.len_episode = len_episode
        self.gamma = gamma
        self.rccc_wrt_max = rccc_wrt_max
        self.prob_remain = prob_remain
        self.deterioration_step = deterioration_step
        self.reward_offset = 1

        # used to speed up data generation
        self.count_mdp = CountMDP(num_groups=1, num_states=self.num_states)

        # Parameters for multiple machines
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_states + 1,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_states + 1,), dtype=np.float32
        )
        self.reward_space = spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32)

        # Initialization
        self.step_counter = 0
        self.episode_rewards = 0
        self.training_rewards = []
        self.reset()

    def seed(self, seed=None):
        return

    def update_mdp(self):
        self.num_groups = np.random.choice(self.range_d)
        self.num_budget = np.random.choice(self.range_k)

    def discretize_budget_proportion(self, budget_proportion: float) -> int:
        """Discretize the budget proportion.

        Args:
            budget_proportion (float): The proportion of budget to use.
        Returns:
            budget_to_use (int): The number of budget to use.

        """
        interval_width = 1 / (self.num_budget + 1)
        budget_to_use = int(budget_proportion / interval_width)
        # deal with the boundary case where proportion is 1 and falls into the last interval
        return min(budget_to_use, self.num_budget)

    def select_action_by_priority(self, imaginary_action):
        # assign the budget to use
        imaginary_action[-1] = 0.5
        zero_indices = np.where(self.observations[: self.num_states] == 0)[0]
        imaginary_action[zero_indices] = 0
        prob_action = imaginary_action / np.sum(imaginary_action)
        # prob_action = softmax(action)
        # convert the action to count action
        count_action = np.zeros_like(prob_action)
        state_count = self.observations[: self.num_states] * self.num_groups
        num_samples = 0
        budget_to_use = self.num_budget
        while budget_to_use > 0 and num_samples < self.num_groups:
            action_idx = np.random.choice(range(self.num_states + 1), p=prob_action)
            if action_idx == self.num_states:
                budget_to_use -= 1
            elif state_count[action_idx] > 0:
                count_action[action_idx] += 1
                state_count[action_idx] -= 1
                budget_to_use -= 1
            num_samples += 1
        return count_action[: self.num_states].astype(int)

    def reset(self, sc_idx: Union[int, list] = 0, deterministic=False):
        """Reset the environment."""
        self.update_mdp()
        state_indices = np.random.choice(range(self.num_states), size=self.num_groups)
        initial_state = np.bincount(state_indices, minlength=self.num_states)
        self.observations = np.append(initial_state, self.num_budget) / self.num_groups
        return self.observations

    def step(self, action: np.array):
        """Take a step in the environment."""
        reward = 0
        count_action = self.select_action_by_priority(action)
        count_state = np.round(
            self.observations[: self.num_states] * self.num_groups
        ).astype(int)

        # simulate the next state
        next_count_state = np.zeros_like(count_state)
        for i in range(self.num_states):
            for j in range(count_state[i]):
                action_idx = 1 if count_action[i] > 0 else 0
                count_action[i] -= 1 if count_action[i] > 0 else 0
                next_state_index = np.random.choice(
                    range(self.num_states),
                    p=self.count_mdp.global_transitions[i, :, action_idx],
                )
                next_count_state[next_state_index] += 1
                reward += (
                    self.count_mdp.global_rewards[i, action_idx, 0] + self.reward_offset
                )
        reward /= self.num_groups
        assert (
            np.sum(next_count_state) == self.num_groups
        ), f"{np.sum(next_count_state)} != {self.num_groups}"

        # register the information
        info = {"reward": reward}
        self.episode_rewards += (self.gamma**self.step_counter) * reward
        self.observations = (
            np.concatenate((next_count_state, [self.num_budget])) / self.num_groups
        )

        # get the done
        self.step_counter += 1
        done = self.step_counter >= self.len_episode
        if done:
            self.training_rewards.append(self.episode_rewards)
            self.episode_rewards = 0
            self.step_counter = 0
            self.reset()
        return self.observations, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
