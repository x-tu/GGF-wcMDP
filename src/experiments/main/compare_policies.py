import copy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from env.mrp_env_rccc import MachineReplacement
from solver.dual_mdp import LPData, build_dlp, extract_dlp, policy_dlp, solve_dlp
from utils.ggf import calculate_ggi_reward

args_rccc = {
    "num_episodes": 1000,
    "len_episode": 50,
    "num_samples": 10,
    "rccc_wrt_max": 0.5,
    "num_arms": 6,
    "num_states": 3,
    "prob_remain": np.linspace(start=0.5, stop=0.9, num=6),
    "mat_type": 1,
    "weight_coefficient": 2,
    "num_steps": 100,
    "ggi": True,
    "encoding_int": False,
    "out_csv_name": "test",
    "alpha": 0.15,
    "discount": 0.95,
    "solve_deterministic": False,
}


def run_mc_simulation(args):
    env_mlp = MachineReplacement(
        num_arms=args["num_arms"],
        num_states=args["num_states"],
        rccc_wrt_max=args["rccc_wrt_max"],
        prob_remain=args["prob_remain"],
        mat_type=args["mat_type"],
        weight_coefficient=args["weight_coefficient"],
        num_steps=args["num_steps"],
        ggi=args["ggi"],
        encoding_int=args["encoding_int"],
        out_csv_name=args["out_csv_name"],
    )
    if args["ggi"]:
        # solve the dual GGF model
        mrp_data = LPData(
            num_arms=args["num_arms"],
            num_states=args["num_states"],
            rccc_wrt_max=args["rccc_wrt_max"],
            prob_remain=args["prob_remain"],
            mat_type=args["mat_type"],
            weights=env_mlp.weights,
            discount=args["discount"],
            encoding_int=True,
        )
        # TODO: replace with functions from ggf_dual.py
        mlp_model = build_dlp(mrp_data)
        dlp_results, mlp_model = solve_dlp(model=mlp_model)
        extract_dlp(model=mlp_model, lp_data=mrp_data)
    env_mlp2 = copy.deepcopy(env_mlp)

    # record the rewards
    ep_rewards_stochastic, ep_rewards_deterministic = [], []
    for ep in tqdm(range(1, args["num_episodes"] + 1)):
        # run LP
        state = env_mlp.reset()
        reward_mlp = 0
        for t in range(args["len_episode"]):
            action = policy_dlp(state, mlp_model, mrp_data, deterministic=True)
            next_observation, reward, done, _ = env_mlp.step(action)
            reward_mlp += args["discount"] ** t * reward
            if done:
                break
            else:
                state = np.zeros(args["num_arms"])
                for n in range(args["num_arms"]):
                    state[n] = int(next_observation[n] * args["num_states"])
        ep_rewards_deterministic.append(
            calculate_ggi_reward(env_mlp.weights, reward_mlp)
        )

        # run LP
        state = env_mlp2.reset()
        reward_mlp2 = 0
        for t in range(args["len_episode"]):
            action = policy_dlp(state, mlp_model, mrp_data, deterministic=False)
            next_observation, reward, done, _ = env_mlp.step(action)
            reward_mlp2 += args["discount"] ** t * reward
            if done:
                break
            else:
                state = np.zeros(args["num_arms"])
                for n in range(args["num_arms"]):
                    state[n] = int(next_observation[n] * args["num_states"])
        ep_rewards_stochastic.append(calculate_ggi_reward(env_mlp.weights, reward_mlp2))
    return ep_rewards_deterministic, ep_rewards_stochastic


if __name__ == "__main__":
    # run the simulation
    ep_rewards1, ep_rewards2 = run_mc_simulation(args=args_rccc)
    avg1 = np.zeros(len(ep_rewards1))
    avg2 = np.zeros(len(ep_rewards2))
    for epoch in range(len(ep_rewards1)):
        avg1[epoch] = np.mean(ep_rewards1[:epoch])
        avg2[epoch] = np.mean(ep_rewards2[:epoch])
    # plot the results
    sns.lineplot(avg1, label="Deterministic")
    sns.lineplot(avg2, label="Stochastic")
    plt.xlabel("Episodes")
    plt.ylabel("Rolling Average Reward")
    plt.title(
        f"Dual-LP ({args_rccc['num_arms']} arms, {args_rccc['num_states']} states)"
    )
    plt.show()
    plt.savefig(
        "results/compare_" + str(datetime.now().strftime("%m_%d_%H_%M_%S")) + ".png"
    )
