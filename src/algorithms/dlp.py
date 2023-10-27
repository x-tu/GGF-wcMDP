import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

from env.mrp_env_rccc import MachineReplacement
from solver.dual_mdp import LPData, build_dlp, extract_dlp, policy_dlp, solve_dlp
from utils.common import DotDict


class DLPAgent:
    """MC simulation with Dual LP solutions for the MRP RCCC problem."""

    def __init__(self, params: DotDict):
        self.params = params
        self.time = {}
        start_time = datetime.now()
        self.env = MachineReplacement(
            num_arms=params.num_groups,
            num_states=params.num_states,
            rccc_wrt_max=params.rccc_wrt_max,
            prob_remain=params.prob_remain,
            mat_type=params.mat_type,
            weight_coefficient=params.weight_coefficient,
            num_steps=params.len_episode,
            ggi=params.ggi,
        )
        self.time["Env"] = (datetime.now() - start_time).total_seconds()
        start_dt_time = datetime.now()
        self.mrp_data = LPData(
            num_arms=params.num_groups,
            num_states=params.num_states,
            rccc_wrt_max=params.rccc_wrt_max,
            prob_remain=params.prob_remain,
            mat_type=params.mat_type,
            weights=self.env.weights,
            discount=params.gamma,
            encoding_int=True,
        )
        self.time["Data_build"] = (datetime.now() - start_dt_time).total_seconds()
        start_build_time = datetime.now()
        mlp_model = build_dlp(self.mrp_data)
        self.time["LP_build"] = (datetime.now() - start_build_time).total_seconds()
        start_slv_time = datetime.now()
        models, self.mlp_model = solve_dlp(model=mlp_model)
        self.time["LP_solve"] = (datetime.now() - start_slv_time).total_seconds()
        start_ext_time = datetime.now()
        extract_dlp(mlp_model, self.mrp_data)
        self.time["LP_extract"] = (datetime.now() - start_ext_time).total_seconds()
        self.time["total"] = (datetime.now() - start_time).total_seconds()
        from pandas import DataFrame as df

        time_df = df.from_dict(self.time, orient="index")
        time_df.to_csv(f"results/time_{params.num_groups}.csv")
        print("Models: ", models)

    def run_mc_dlp(self, initial_states: list = None):
        """Run the MC simulation.

        Notice: Random seeds may vary across different python versions or devices.
            We can force use a given initial state list if needed.

        Args:
            initial_states: list of initial states to run the simulation
        """

        episode_rewards = []
        for ep in tqdm(range(self.params.num_episodes)):
            # run LP
            ep_rewards = []
            for n in range(self.params.num_samples):
                init_state = random.randint(0, self.params.num_states - 1)
                # initial_state = initial_states[ep] if initial_states else 0
                state = self.env.reset(initial_state=init_state)
                total_reward = 0
                for t in range(self.params.len_episode):
                    action = policy_dlp(state, self.mlp_model, self.mrp_data)
                    next_observation, reward, done, _ = self.env.step(action)
                    total_reward += (1 - done) * self.mrp_data.discount ** t * reward
                    state = (next_observation * self.env.num_states).astype(int)
                ep_rewards.append(total_reward)
            # get the expected rewards by averaging over samples, and then sort
            rewards_sorted = np.sort(np.mean(ep_rewards, axis=0))
            episode_rewards.append(np.dot(self.env.weights, rewards_sorted))
        return episode_rewards
