import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from algorithms.dqn_mrp import ODQNAgent, RDQNAgent
from algorithms.policy_iteration import PIAgent
from algorithms.tabular_q import QAgent
from env.mrp_env_rccc import MachineReplacement
from experiments.configs.config_shared import prs_parser_setup
from solver.dual_mdp import LPData, build_dlp, extract_dlp, policy_dlp, solve_dlp
from solver.ggf_dual import get_policy as get_policy_ggf, solve_ggf
from solver.momdp import get_policy, solve_mrp


def calculate_ggi_reward(weights, n_rewards):
    # assign the largest value to the smallest weight
    weights = sorted(weights, reverse=True)
    n_rewards = sorted(n_rewards, reverse=False)
    ggi_reward = np.dot(weights, n_rewards)
    return ggi_reward


def run_mc_simulation(
    args,
    num_episodes=1000,
    len_episode=100,
    alpha=0.15,
    epsilon=0.3,
    decaying_factor=0.95,
    gamma=0.95,
):
    # Replacement Cost Constant Coefficient (RCCC) w.r.t the maximum cost in passive mode
    rccc_wrt_max = 0.5
    # Number of arms
    num_arms = 3
    num_states = 3
    # Probability of remaining in the same state for each machine in each state
    prob_remain = np.linspace(start=0.5, stop=0.9, num=num_arms)
    # The type of deterioration transition matrix
    mat_type = 1

    env_mlp = MachineReplacement(
        num_arms=num_arms,
        num_states=num_states,
        rccc_wrt_max=rccc_wrt_max,
        prob_remain=prob_remain,
        mat_type=mat_type,
        weight_coefficient=2,
        num_steps=100,
        ggi=args.ggi,
        encoding_int=False,
        out_csv_name="test",
    )
    solve_deterministic = False
    if args.ggi:
        # solve the dual GGF model
        discount = 0.95
        mrp_data = LPData(
            num_arms,
            num_states,
            rccc_wrt_max,
            prob_remain,
            mat_type,
            env_mlp.weights,
            discount,
            encoding_int=True,
        )
        # TODO: replace with functions from ggf_dual.py
        # _, mlp_model = solve_ggf(
        #     mrp_data, solve_deterministic=solve_deterministic
        # )
        mlp_model = build_dlp(mrp_data)
        dlp_results, mlp_model = solve_dlp(model=mlp_model)
        extract_dlp(model=mlp_model, lp_data=mrp_data)
    # else:
    # solve the MOMDP model (summed reward)
    # _, mlp_model = solve_mrp(
    #     env_mlp.mrp_data, solve_deterministic=solve_deterministic
    # )

    # initialize the environment for Q Learning
    env_ql = MachineReplacement(
        num_arms=num_arms,
        num_states=num_states,
        rccc_wrt_max=rccc_wrt_max,
        prob_remain=prob_remain,
        mat_type=mat_type,
        weight_coefficient=2,
        num_steps=100,
        ggi=args.ggi,
        encoding_int=True,
        out_csv_name="test",
    )
    agent = QAgent(
        env=env_ql,
        num_states=env_ql.num_states,
        num_actions=env_ql.num_actions,
        num_groups=env_ql.num_arms,
        alpha=alpha,
        epsilon=epsilon,
        decaying_factor=decaying_factor,
        gamma=gamma,
    )
    agent.lr_decay_schedule = np.linspace(alpha, 0, num_episodes)

    # initialize the environment for Policy Iteration
    env_pi = copy.deepcopy(env_ql)
    params = {
        "n_group": args.n_group,
        "n_state": args.n_state,
        "n_action": args.n_action,
        "ggi": False,  # TODO: fix the GGF-PI not converging issue (multiple optimal policies)
    }
    policy_agent = PIAgent(params, gamma=0.99, theta=1e-10)
    # run policy iteration
    V, pi = policy_agent.run_policy_iteration()
    policy_pi = {s: pi(s) for s in range(policy_agent.n_state)}

    # initialize the environment for DQN
    env_dqn = copy.deepcopy(env_mlp)
    # env_dqn.encoding_int = False
    agent_dqn = RDQNAgent(
        mrp_data, discount, args.ggi, mrp_data.weights, l_rate=1e-3, h_size=128
    )

    # record the rewards
    episode_rewards_mlp, episode_rewards_ql, episode_rewards_pi = [], [], []
    episode_rewards_dqn = []
    for ep in range(1, num_episodes + 1):
        # run LP
        state = env_mlp.reset()
        reward_mlp = 0
        for t in range(len_episode):
            action = policy_dlp(state, mlp_model, mrp_data)
            next_observation, reward, done, _ = env_mlp.step(action)
            reward_mlp += discount ** t * reward
            if done:
                break
            else:
                state = np.zeros(num_arms)
                for n in range(num_arms):
                    state[n] = int(next_observation[n] * num_states)
        episode_rewards_mlp.append(calculate_ggi_reward(env_mlp.weights, reward_mlp))
        #     if args.ggi:
        #         action = get_policy_ggf(state, mlp_model, mrp_data)
        #     else:
        #         action = get_policy(state, mlp_model, mrp_data)
        #     next_state, reward, done, _ = env_mlp.step(action)
        #     reward_mlp += gamma ** t * reward
        #     state = next_state
        # if args.ggi:
        #     # calculate the GGI reward
        #     episode_rewards_mlp.append(
        #         calculate_ggi_reward(env_mlp.weights, reward_mlp)
        #     )
        # else:
        #     episode_rewards_mlp.append(reward_mlp)

        # run Q Learning
        state = env_ql.reset()
        reward_ql = 0
        for t in range(len_episode):
            action = agent.get_action(state, t)
            state_next, reward, done, _ = env_ql.step(action)
            agent.update_q_function(state, action, reward, state_next)
            state = state_next
            reward_ql += (gamma ** t) * reward
        # We use a decaying epsilon-greedy method
        if agent.epsilon > 0.001:
            agent.epsilon = agent.epsilon * agent.decaying_factor
        agent.alpha = agent.lr_decay_schedule[ep - 1]
        if args.ggi:
            # calculate the GGI reward
            episode_rewards_ql.append(calculate_ggi_reward(env_ql.weights, reward_ql))
        else:
            episode_rewards_ql.append(reward_ql)

        # run policy iteration
        state = env_pi.reset()
        reward_pi = 0
        for t in range(len_episode):
            action = policy_pi[state]
            next_state, reward, done, _ = env_pi.step(action)
            reward_pi += gamma ** t * reward
            state = next_state
        if args.ggi:
            # calculate the GGI reward
            episode_rewards_pi.append(calculate_ggi_reward(env_pi.weights, reward_pi))
        else:
            episode_rewards_pi.append(reward_pi)

        # run DQN
        observation = env_dqn.reset()
        reward = [0] * num_arms
        reward_dqn = 0
        for t in range(len_episode):
            action = agent_dqn.act(observation, reward)
            next_observation, reward, done, _ = env_dqn.step(action)
            reward_dqn += (gamma ** t) * reward
            if done:
                break
            else:
                agent_dqn.update(observation, action, reward, next_observation)
                observation = next_observation
        if args.ggi:
            # calculate the GGI reward
            episode_rewards_dqn.append(
                calculate_ggi_reward(env_dqn.weights, reward_dqn)
            )
        else:
            episode_rewards_dqn.append(reward_dqn)
    return (
        episode_rewards_mlp,
        episode_rewards_ql,
        episode_rewards_pi,
        episode_rewards_dqn,
    )


if __name__ == "__main__":
    # configuration parameters
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Experiments with MRP RCCC.""",
    )
    prs = prs_parser_setup(prs)
    args_rccc, _ = prs.parse_known_args()

    # run the simulation
    ep_rewards_mlp, ep_rewards_ql, ep_rewards_pi, ep_rewards_dqn = run_mc_simulation(
        args=args_rccc
    )

    # plot the results
    sns.lineplot(ep_rewards_mlp, label="MLP")
    sns.lineplot(ep_rewards_pi, label="Policy Iteration")
    sns.lineplot(ep_rewards_ql, label="Q Learning")
    sns.lineplot(ep_rewards_dqn, label="DQN")
    plt.show()
