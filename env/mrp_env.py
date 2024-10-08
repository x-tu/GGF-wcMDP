# temporarily disable the warnings
import warnings

warnings.filterwarnings("ignore")

import gym
import numpy as np
import pandas as pd
from gym import spaces

from utils.mrp import MRPData


class MachineReplace(gym.Env):
    def __init__(
        self,
        n_group: int,
        n_state: int,
        n_action: int,
        out_csv_name: str = None,
        ggi: bool = False,
        init_state: int = 0,
        save_mode="append",
        encoding_int: bool = True,
    ):
        self.ggi = ggi
        self.mrp_data = MRPData(n_group=n_group, n_state=n_state, n_action=n_action)
        self.actions = self.mrp_data.tuple_list_a
        self.states = self.mrp_data.tuple_list_s
        self.num_actions = len(self.actions)
        self.num_arms = n_group
        self.weights = self.mrp_data.weight
        self.encoding_int = encoding_int

        # Define the observation and action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_space = spaces.Discrete(n_group)
        self.observation_space = spaces.Discrete(len(self.states))

        # Initialization
        self.save_mode = save_mode
        self.out_csv_name = out_csv_name
        self.metrics = []
        self.state = 0  # index of state
        self.run = 0
        self.step_counter = 0
        self.episode_length = 2000
        self.init_state = init_state
        self.reset()

    def seed(self, seed=None):
        return

    def reset(self):
        """ Reset the environment to the initial state.

        Returns:
            state (np.array): the initial state of the environment.

        """
        # set initial state to 0 (all machines are new)
        self.step_counter = 0
        self.run += 1
        self.metrics = []

        self.state = self.init_state
        if self.encoding_int:
            return self.state
        else:
            return np.array(self.states[self.state])

    def step(self, action):
        """ Take a step in the environment.

        Args:
            action (`int`): the action to take in the environment.

        Returns:
            state (`np.array`): the state of the environment after taking the action.
            reward (`float`): the reward for taking the action.
            done (`bool`): whether the episode is done.
            info (`dict`): additional information about the environment.

        """
        # update next state according to current state
        next_state_prob = self.mrp_data.bigT[self.state, :, action]
        # get the state
        self.state = np.random.choice(
            np.arange(len(next_state_prob)), p=next_state_prob
        )
        # get the reward
        reward = -self.mrp_data.bigC[self.state, action, :] + 110
        # get the done
        done = self.step_counter >= self.episode_length
        # register the information
        info = {f"reward_{n}": reward[n] for n in range(len(reward))}
        self.metrics.append(info)
        if not self.ggi:
            reward = sum(reward)
        self.step_counter += 1
        if done:
            self.save_csv()
        if self.encoding_int:
            return self.state, reward, done, info
        else:
            return np.array(self.states[self.state]), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def save_csv(self) -> None:
        """ Used to save the metrics in a csv file.

        Args:
            out_csv_name: the name of the csv file
            run: indicate the  when generating multiple files

        Returns:
            None

        """
        if self.out_csv_name is not None:
            if self.save_mode == "write":
                df = pd.DataFrame(self.metrics)
                df.to_csv(
                    self.out_csv_name + "_run{}".format(self.run) + ".csv", index=False
                )
            elif self.save_mode == "append":
                if self.run == 1:
                    df = pd.DataFrame(self.metrics)
                    df.to_csv(self.out_csv_name + ".csv", header=True, index=False)
                else:
                    df = pd.DataFrame(self.metrics)
                    df.to_csv(
                        self.out_csv_name + ".csv", mode="a", header=False, index=False
                    )
            else:
                raise TypeError("Invalid save mode")
