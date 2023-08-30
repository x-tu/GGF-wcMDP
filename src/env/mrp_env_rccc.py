# temporarily disable the warnings
import warnings

import gym
import numpy as np
import pandas as pd

from utils.params_mrp import CostReward, FairWeight, MarkovChain

warnings.filterwarnings("ignore")


class MachineReplacement(gym.Env):
    def __init__(
        self,
        num_arms: int,
        num_states: int,
        rccc_wrt_max: float,
        prob_remain,
        mat_type: int,
        weight_coefficient: int,
        num_steps: int,
        out_csv_name: str,
    ):
        super(gym.Env, self).__init__()
        # Parameters
        self.num_arms = num_arms
        self.num_states = num_states
        self.num_steps = num_steps
        self.num_actions = self.num_arms + 1
        self.out_csv_name = out_csv_name

        # Basic Functions
        rew_class = CostReward(num_states, num_arms, rccc_wrt_max)
        self.rewards = rew_class.rewards
        dyn_class = MarkovChain(num_states, num_arms, prob_remain, mat_type)
        self.transitions = dyn_class.transitions
        wgh_class = FairWeight(num_arms, weight_coefficient)
        self.weights = wgh_class.weights

        # Initialization
        self.n_runs = 0
        self.step_counter = 0
        self.reward_info = []
        self.observations = self.reset()

    def seed(self, seed=None):
        return

    def reset(self):
        self.n_runs += 1
        self.step_counter = 0
        self.reward_info = []
        return np.zeros(self.num_arms, dtype=int)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (`int`): the action to take in the environment.

        Returns:
            state (`np.array`): the state of the environment after taking the action.
            reward (`float`): the reward for taking the action.
            done (`bool`): whether the episode is done.
            info (`dict`): additional information about the environment.

        """
        # convert the observation vector into the state vector
        state_list = self.observations * self.num_states
        next_state_list = np.copy(state_list)
        # convert the action integer into the action list
        action_list = np.zeros(self.num_arms, dtype=int)
        # get the vector of rewards
        reward_list = np.zeros(self.num_arms)
        if action > 0:
            action_list[action - 1] = 1
        for n in range(self.num_arms):
            next_state_prob = self.transitions[int(state_list[n]), :, n, action_list[n]]
            # get the state
            next_state_list[n] = np.random.choice(
                np.arange(len(next_state_prob)), p=next_state_prob
            )
            # get the reward
            reward_list[n] = self.rewards[int(state_list[n]), n, action_list[n]]
        self.observations = next_state_list / self.num_states
        # get the done
        done = self.step_counter >= self.num_steps
        # register the information
        info = {f"reward_{n}": reward_list[n] for n in range(self.num_arms)}
        self.reward_info.append(info)
        self.step_counter += 1
        if done:
            self.save_csv()
        return self.observations, reward_list, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def save_csv(self) -> None:
        if self.out_csv_name is not None:
            if self.n_runs == 1:
                df = pd.DataFrame(self.reward_info)
                df.to_csv(self.out_csv_name + ".csv", header=True, index=False)
            else:
                df = pd.DataFrame(self.reward_info)
                df.to_csv(
                    self.out_csv_name + ".csv", mode="a", header=False, index=False
                )
