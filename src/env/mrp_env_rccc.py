# temporarily disable the warnings
import warnings

import gym
import numpy as np
from gym import spaces

from utils.encoding import state_int_index_to_vector, state_vector_to_int_index
from utils.mrp import MRPData

warnings.filterwarnings("ignore")


class MachineReplacement(gym.Env):
    """A gym environment for the machine replacement problem."""

    def __init__(
        self,
        num_groups: int,
        num_states: int,
        num_actions: int,
        num_steps: int,
        encoding_int: bool = False,
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
        # decide whether to use integer encoding for state and action
        self.encoding_int = encoding_int

        # get data for all machines
        self.mrp_data = MRPData(
            num_groups=num_groups,
            num_states=num_states,
            num_actions=num_actions,
            rccc_wrt_max=rccc_wrt_max,
            prob_remain=prob_remain,
            deterioration_step=deterioration_step,
        )

        # Parameters for multiple machines
        self.observation_space = spaces.Discrete(self.mrp_data.num_global_states)
        self.action_space = spaces.Discrete(self.mrp_data.num_global_actions)
        self.reward_space = spaces.Discrete(self.mrp_data.num_groups)

        # Initialization
        self.n_runs = 0
        self.step_counter = 0
        self.reward_info = []
        self.observations = self.reset()

    def seed(self, seed=None):
        return

    def reset(self, initial_state: int = 0, normalize: bool = False):
        """Reset the environment.

        Args:
            initial_state (`int`): the initial state of the environment.
            normalize (`bool`): whether to normalize the state vector.
        """

        self.n_runs += 1
        self.step_counter = 0
        self.reward_info = []
        # set initial state as given
        if self.encoding_int:
            return initial_state
        state_vector = state_int_index_to_vector(
            initial_state, self.num_groups, self.num_states
        )
        if normalize:
            return state_vector / self.num_states
        return state_vector

    def step(self, action: int):
        """Take a step in the environment.

        Args:
            action (`int`): the action to take in the environment.

        Returns:
            state (`np.array`): the state of the environment after taking the action.
            reward (`float`): the reward for taking the action.
            done (`bool`): whether the episode is done.
            info (`dict`): additional information about the environment.
        """

        if self.encoding_int & isinstance(self.observations, int):
            state = self.observations
        else:
            # convert the observation vector into the state vector
            state_vector = self.observations * self.num_states
            state = state_vector_to_int_index(
                state_vector=state_vector, num_states=self.num_states
            )
        # get next state
        next_state_prob = self.mrp_data.global_transitions[state, :, action]
        next_state = np.random.choice(
            np.arange(self.observation_space.n), p=next_state_prob
        )
        # get the reward
        reward_list = self.mrp_data.global_costs[state, action, :]
        # get the done
        done = self.step_counter >= self.num_steps
        # register the information
        info = {f"reward_{n}": reward_list[n] for n in range(self.num_groups)}
        # update the counter
        self.step_counter += 1
        if self.encoding_int:
            self.observations = next_state
        else:
            next_state_list = state_int_index_to_vector(
                state_int_index=next_state,
                num_groups=self.num_groups,
                num_states=self.num_states,
            )
            self.observations = next_state_list / self.num_states
        return self.observations, reward_list, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
