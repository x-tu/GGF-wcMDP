import numpy as np
from tqdm import tqdm

from env.mrp_env_rccc import MachineReplacement
from solver.dual_mdp import LPData, build_dlp, policy_dlp, solve_dlp
from utils.common import DotDict


class DLPAgent:
    """MC simulation with Dual LP solutions for the MRP RCCC problem."""

    def __init__(self, params: DotDict):
        self.params = params
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
        mlp_model = build_dlp(self.mrp_data)
        _, self.mlp_model = solve_dlp(model=mlp_model)

    def run_mc_dlp(self, initial_state: int = 0):
        """Run the MC simulation."""

        episode_rewards = []
        for _ in tqdm(range(self.params.num_episodes)):
            # run LP
            state = self.env.reset(initial_state=initial_state)
            total_reward = 0
            for t in range(self.params.len_episode):
                action = policy_dlp(state, self.mlp_model, self.mrp_data)
                next_observation, reward, done, _ = self.env.step(action)
                total_reward += (1 - done) * self.mrp_data.discount ** t * reward
                state = (next_observation * self.env.num_states).astype(int)
            rewards_sorted = np.sort(total_reward)
            episode_rewards.append(np.dot(self.env.weights, rewards_sorted))
        return episode_rewards
