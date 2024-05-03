# temporarily disable the warnings
import warnings
from typing import Union

import gym
import numpy as np
from gym import spaces

from utils.count import CountMDP, count_to_normal

warnings.filterwarnings("ignore")


class CountMDPEnv(gym.Env):
    """A gym environment for the machine replacement problem."""

    def __init__(
            self,
            num_groups: int,
            num_states: int,
            num_actions: int,
            len_episode: int = 300,
            gamma: float = 0.95,
            rccc_wrt_max: float = 1.5,
            prob_remain: float = 0.8,
            deterioration_step: int = 1,
    ):
        super(gym.Env, self).__init__()

        # parameters
        self.num_groups = num_groups
        self.num_states = num_states
        self.num_actions = num_actions
        self.len_episode = len_episode
        self.gamma = gamma
        self.count_mdp = CountMDP(
            num_groups=num_groups,
            num_states=num_states,
            num_actions=num_actions,
            rccc_wrt_max=rccc_wrt_max,
            prob_remain=prob_remain,
            deterioration_step=deterioration_step,
        )
        reward_copy = self.count_mdp.count_rewards.copy()
        reward_copy[reward_copy == -1e6] = 0
        self.reward_offset = reward_copy.min()

        # Parameters for multiple machines
        self.observation_space = spaces.Discrete(self.count_mdp.num_count_states)
        self.action_space = spaces.Discrete(self.count_mdp.num_groups+1)
        self.reward_space = spaces.Box(low=0, high=-self.reward_offset, shape=(1,), dtype=np.float32)

        # Initialization
        self.step_counter = 0
        self.episode_rewards = 0
        self.training_rewards = []
        self.reward_info = []
        self.observations = self.reset()

    def seed(self, seed=None):
        return

    def reset(self, sc_idx: Union[int, list] = 0, deterministic=False):
        """Reset the environment."""
        sc_idx = np.random.randint(0, len(self.count_mdp.count_states))
        return sc_idx

    def step(self, action: int):
        """Take a step in the environment."""

        ac_idx = self.count_mdp.count_env_idx_mapping[(self.observations, action)]
        # get next state
        next_state_prob = self.count_mdp.count_transitions[self.observations, :, ac_idx]
        next_state = np.random.choice(
            np.arange(self.observation_space.n), p=next_state_prob
        )
        # get the reward
        reward = self.count_mdp.count_rewards[self.observations, ac_idx] - self.reward_offset
        # get the done
        self.step_counter += 1
        done = self.step_counter >= self.len_episode
        if done:
            num_groups = self.count_mdp.num_groups
            # print("ep r:", (-self.reward_offset * (1/(1-self.gamma)) - self.episode_rewards)/num_groups)
            rescaled_ep_reward = (-self.reward_offset * (1/(1-self.gamma)) - self.episode_rewards)/num_groups
            self.training_rewards.append(rescaled_ep_reward)
            self.episode_rewards = 0
            self.step_counter = 0
        # register the information
        info = {"reward": reward}
        self.episode_rewards += (self.gamma ** self.step_counter) * reward
        self.observations = next_state
        return self.observations, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
