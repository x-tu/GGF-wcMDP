import numpy as np

from algorithms.policy_iteration import PIAgent
from algorithms.tabular_q import QAgent
from env.mrp_env import MachineReplace
from solver.momdp import get_policy, solve_mrp


def run_mc_simulation(
    args,
    num_episodes=500,
    len_episode=1000,
    alpha=0.15,
    epsilon=0.3,
    decaying_factor=0.95,
    gamma=0.99,
):
    # initialize the environment for LP
    env_mlp = MachineReplace(
        n_group=args.n_group,
        n_state=args.n_state,
        n_action=args.n_action,
        init_state=0,
        ggi=args.ggi,
    )
    # solve the LP model
    results, mlp_model = solve_mrp(env_mlp.mrp_data)

    # initialize the environment for Q Learning
    env_ql = MachineReplace(
        n_group=args.n_group,
        n_state=args.n_state,
        n_action=args.n_action,
        init_state=0,
        ggi=args.ggi,
    )
    agent = QAgent(
        env=env_ql,
        num_states=env_ql.observation_space.n,
        num_actions=env_ql.action_space.n,
        num_groups=env_ql.mrp_data.n_group,
        alpha=alpha,
        epsilon=epsilon,
        decaying_factor=decaying_factor,
        gamma=gamma,
    )
    agent.lr_decay_schedule = np.linspace(alpha, 0, num_episodes)

    # initialize the environment for Policy Iteration
    env_pi = MachineReplace(
        n_group=args.n_group,
        n_state=args.n_state,
        n_action=args.n_action,
        init_state=0,
        ggi=args.ggi,
    )
    params = {
        "n_group": args.n_group,
        "n_state": args.n_state,
        "n_action": args.n_action,
        "ggi": args.ggi,
    }
    policy_agent = PIAgent(params, gamma=0.99, theta=1e-10)
    # run policy iteration
    V, pi = policy_agent.run_policy_iteration()
    policy_pi = {s: pi(s) for s in range(policy_agent.n_state)}

    # record the rewards
    episode_rewards_mlp, episode_rewards_ql, episode_rewards_pi = [], [], []
    for ep in range(1, num_episodes + 1):
        # run LP
        state = env_mlp.reset()
        reward_mlp = 0
        for t in range(len_episode):
            action = get_policy(state, mlp_model, env_mlp.mrp_data)
            next_state, reward, done, _ = env_mlp.step(action)
            reward_mlp += gamma ** t * reward
            state = next_state
        episode_rewards_mlp.append(reward_mlp)

        # run Q Learning
        state = env_ql.reset()
        reward_ql = 0
        for t in range(len_episode):
            action = agent.get_action(state)
            state_next, reward, done, _ = env_ql.step(action)
            agent.update_q_function(state, action, reward, state_next)
            state = state_next
            reward_ql += (gamma ** t) * reward
        # We use a decaying epsilon-greedy method
        if agent.epsilon > 0.001:
            agent.epsilon = agent.epsilon * agent.decaying_factor
        agent.alpha = agent.lr_decay_schedule[ep - 1]
        episode_rewards_ql.append(reward_ql)

        # run policy iteration
        state = env_pi.reset()
        reward_pi = 0
        for t in range(len_episode):
            action = policy_pi[state]
            next_state, reward, done, _ = env_pi.step(action)
            reward_pi += gamma ** t * reward
            state = next_state
        episode_rewards_pi.append(reward_pi)
    return episode_rewards_mlp, episode_rewards_ql, episode_rewards_pi


# run the script as main
if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from experiments.configs.config_shared import prs_parser_setup

    # configuration parameters
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Tabular Q Learning.""",
    )
    prs = prs_parser_setup(prs)
    args, _ = prs.parse_known_args()

    episode_rewards_mlp, episode_rewards_ql, episode_rewards_pi = run_mc_simulation(
        args=args
    )
    sns.lineplot(episode_rewards_mlp, label="MLP")
    sns.lineplot(episode_rewards_ql, label="Q Learning")
    sns.lineplot(episode_rewards_pi, label="Policy Iteration")
    plt.show()
