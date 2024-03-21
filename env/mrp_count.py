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
            num_steps: int,
            rccc_wrt_max: float = 1.5,
            prob_remain: float = 0.8,
            deterioration_step: int = 1,
    ):
        super(gym.Env, self).__init__()

        # parameters
        self.num_groups = num_groups
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.count_mdp = CountMDP(
            num_groups=num_groups,
            num_states=num_states,
            num_actions=num_actions,
            rccc_wrt_max=rccc_wrt_max,
            prob_remain=prob_remain,
            deterioration_step=deterioration_step,
        )
        # Parameters for multiple machines
        self.observation_space = spaces.Discrete(self.count_mdp.num_count_states)
        self.action_space = spaces.Discrete(self.count_mdp.num_groups+1)
        self.reward_space = spaces.Box(low=-99, high=0, shape=(1,), dtype=np.float32)

        # Initialization
        self.step_counter = 0
        self.episode_rewards = 0
        self.training_rewards = []
        self.reward_info = []
        self.observations = self.reset()

    def seed(self, seed=None):
        return

    def reset(self, s_idx: Union[int, list] = 0, deterministic=False):
        """Reset the environment."""
        self.training_rewards.append(self.episode_rewards)
        self.step_counter = 0
        self.episode_rewards = 0
        self.reward_info = []
        return s_idx/len(self.count_mdp.count_states)
        # return np.array(self.count_mdp.count_states[s_idx])/len(self.count_mdp.count_states)

    def step(self, action: int):
        """Take a step in the environment."""

        s_idx = int(self.observations * len(self.count_mdp.count_states))
        sc_idx = self.count_mdp.sn_idx_to_s_idx_mapping[s_idx]
        sc_vec = self.count_mdp.s_idx_to_x_mapping[str(sc_idx)]
        s_vec = count_to_normal(sc_vec)
        a_vec = [0] * self.num_groups
        if action > 0:
            a_vec[action-1] = 1
        ac_vec = np.array(self.count_mdp.action_normal_to_count(a_vec=a_vec, s_vec=s_vec))
        a_idx = self.count_mdp.ac_to_idx_mapping[str(ac_vec)]

        # get next state
        next_state_prob = self.count_mdp.count_transitions[s_idx, :, a_idx]
        next_state = np.random.choice(
            np.arange(self.observation_space.n), p=next_state_prob
        )
        # get the reward
        reward = self.count_mdp.count_rewards[s_idx, a_idx]
        # get the done
        self.step_counter += 1
        done = self.step_counter >= self.num_steps
        # register the information
        info = {f"reward": reward}
        self.episode_rewards += reward * 0.95 ** self.step_counter
        self.observations = next_state / len(self.count_mdp.count_states)
        return self.observations, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
