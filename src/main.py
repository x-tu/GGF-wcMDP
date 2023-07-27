from params_mrp import get_state_list
from env_mrp import MachineReplacement
from dqn import DQNAgent
from dual_mdp import LPData, build_lp, solve_lp, extract_lp, policy_lp
from muob_mdp import build_mlp, solve_mlp, extract_mlp, policy_mlp
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':

    # Number of arms
    num_arms = 4
    # Number of states for each arm
    num_states = 4
    # Replacement Cost Constant Coefficient (RCCC) w.r.t the maximum cost in passive mode
    rccc_wrt_max = 0.75
    # Probability of remaining in the same state for each machine in each state
    prob_remain = 0.5 * np.ones(num_arms)
    # The type of deterioration transition matrix
    mat_type = 1
    # The discount factor
    discount = 0.95
    # The fair weight coefficient
    weight_coefficient = 1
    # The number of time steps (for higher discount factor should be set higher)
    num_steps = 200
    # The number of learning episodes
    num_episodes = 200

    # ----------------------------------- DUAL MO-MDP Setup -----------------------------------

    data_mrp = LPData(num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, discount)
    lp_model = build_lp(data_mrp)
    lp_results, lp_model = solve_lp(model=lp_model)
    extract_lp(model=lp_model, lp_data=data_mrp)
    opt_csv_name = "results_opt_ggi"
    env_lp = MachineReplacement(
        num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, num_steps, opt_csv_name
    )
    opt_rewards = []

    # ----------------------------------- Multi-Objective Setup -----------------------------------

    data_mrp = LPData(num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, discount)
    mlp_model = build_mlp(data_mrp)
    mlp_results, mlp_model = solve_mlp(model=mlp_model)
    extract_mlp(model=mlp_model, data=data_mrp)
    mlp_csv_name = "results_mlp_ggi"
    env_mlp = MachineReplacement(
        num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, num_steps, mlp_csv_name
    )
    mlp_rewards = []

    # ----------------------------------- DQN Setup -----------------------------------

    initial_lr = 1e-3
    dqn_csv_name = "results_dqn_ggi"
    env_dqn = MachineReplacement(
        num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, num_steps, dqn_csv_name
    )
    agent = DQNAgent(env_dqn.num_arms, env_dqn.num_actions, discount, initial_lr)
    dqn_rewards = []

    # ----------------------------------- Monte-Carlo Simulations -----------------------------------

    for i_episode in range(num_episodes):

        print("Episode: " + str(i_episode))

        state = env_lp.reset()
        opt_reward = 0
        for t in range(num_steps):
            action = policy_lp(state, lp_model, data_mrp)
            next_observation, reward, done, _ = env_lp.step(action)
            state = np.zeros(num_arms)
            for n in range(num_arms):
                state[n] = int(next_observation[n] * num_states)
            opt_reward += discount ** t * reward
            if done:
                break
        opt_rewards.append(opt_reward)

        state = env_mlp.reset()
        mlp_reward = 0
        for t in range(num_steps):
            action = policy_mlp(state, mlp_model, data_mrp)
            next_observation, reward, done, _ = env_mlp.step(action)
            state = np.zeros(num_arms)
            for n in range(num_arms):
                state[n] = int(next_observation[n] * num_states)
            mlp_reward += discount ** t * reward
            if done:
                break
        mlp_rewards.append(mlp_reward)

        observation = env_dqn.reset()
        dqn_reward = 0
        for t in range(num_steps):
            action = agent.act(observation)
            next_observation, reward, done, _ = env_dqn.step(action)
            dqn_reward += discount**t * reward
            agent.update(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                break
        dqn_rewards.append(dqn_reward)

    opt_rewards = np.array(opt_rewards)
    mlp_rewards = np.array(mlp_rewards)
    dqn_rewards = np.array(dqn_rewards)
    rewards_opt = opt_rewards.copy()
    rewards_mlp = mlp_rewards.copy()
    rewards_dqn = dqn_rewards.copy()
    for i in range(num_episodes):
        rewards_opt[i] = np.mean(opt_rewards[0:i])
        rewards_mlp[i] = np.mean(mlp_rewards[0:i])
        rewards_dqn[i] = np.mean(dqn_rewards[0:i])

    # ----------------------------------------- Results -----------------------------------------

    states_list = get_state_list(num_states, num_arms)
    for state in states_list:
        observation = np.array(state) / num_states
        action = agent.act(observation)
        print("State: {} -> DQN Action: {}".format(str(state), str(action)))

    fig, ax = plt.subplots()
    ax.plot(range(len(rewards_dqn)), rewards_dqn, label="DQN")
    ax.plot(range(len(rewards_opt)), rewards_opt, label="OPT")
    ax.plot(range(len(rewards_mlp)), rewards_mlp, label="MLP")
    ax.set(xlabel='Episodes', ylabel='Discounted Reward',
           title='Learning Curve')
    ax.grid()
    plt.legend()
    plt.show()
