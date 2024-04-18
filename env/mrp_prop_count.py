# temporarily disable the warnings
import warnings
from typing import Union

import gym
import numpy as np
from gym import spaces

from utils.count import CountMDP, softmax

warnings.filterwarnings("ignore")


class PropCountMDPEnv(gym.Env):
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
        self.count_mdp_pool = {}
        for num_group in self.range_d:
            self.count_mdp_pool[num_group] = CountMDP(
                num_groups=num_group, num_states=self.num_states
            )
        self.count_mdp = None

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
        self.count_mdp = self.count_mdp_pool[self.num_groups]

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

    def select_action_by_priority(self, composed_action):
        # assign the budget to use
        budget_to_use = self.discretize_budget_proportion(composed_action[-1])

        # forbid taking actions if no machines
        action = softmax(composed_action[: self.num_states])
        zero_indices = np.where(self.observations[: self.num_states] == 0)[0]
        action[zero_indices] = 0
        prob_action = action / np.sum(action)

        # convert the action to count action
        count_action = np.zeros_like(action)
        state_count = self.observations[: self.num_states] * self.num_groups
        while budget_to_use > 0:
            action_idx = np.random.choice(range(self.num_states), p=prob_action)
            if state_count[action_idx] > 0:
                count_action[action_idx] += 1
                state_count[action_idx] -= 1
                budget_to_use -= 1
        return count_action.astype(int)

    def reset(self, sc_idx: Union[int, list] = 0, deterministic=False):
        """Reset the environment."""
        self.update_mdp()
        sc_idx = np.random.randint(0, len(self.count_mdp.count_states))
        self.observations = self.count_mdp.count_state_props[sc_idx]
        return self.observations

    def step(self, action: np.array):
        """Take a step in the environment."""
        # get action idx
        count_action = self.select_action_by_priority(action)
        action_idx = self.count_mdp.ac_to_idx_mapping[str(count_action)]
        # get state idx
        count_state = self.observations[: self.num_states] * self.num_groups
        state_idx = self.count_mdp.mapping_x_to_idx[str(list(count_state.astype(int)))]
        # transit to the next state
        next_state_prob = self.count_mdp.count_transitions[state_idx, :, action_idx]
        next_state_idx = np.random.choice(
            np.arange(len(next_state_prob)), p=next_state_prob
        )
        next_state = self.count_mdp.count_states[next_state_idx]
        # get the reward
        reward = (
            self.count_mdp.count_rewards[state_idx, action_idx] / self.num_groups
            + self.reward_offset
        )

        # register the information
        info = {"reward": reward}
        self.episode_rewards += (self.gamma**self.step_counter) * reward
        # TODO: can be optimized
        self.observations = np.array(next_state + [self.num_budget]) / self.num_groups

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
