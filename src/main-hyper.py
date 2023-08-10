import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import os
from params_mrp import FairWeight
from env_mrp import MachineReplacement
from dqn_mrp import RDQNAgent, ODQNAgent
from dual_mdp import LPData, build_dlp, solve_dlp, extract_dlp, policy_dlp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':

    # Number of arms
    num_arms = 3
    # Number of states for each arm
    num_states = 3
    # Replacement Cost Constant Coefficient (RCCC) w.r.t the maximum cost in passive mode
    rccc_wrt_max = 0.5
    # Probability of remaining in the same state for each machine in each state
    prob_remain = np.linspace(start=0.5, stop=0.9, num=num_arms)
    # The type of deterioration transition matrix
    mat_type = 1
    # The discount factor
    discount = 0.95
    # Whether to consider GGI case or the average model:
    ggi_flag = False
    # The fair weight coefficient
    if ggi_flag:
        weight_coefficient = 1
    else:
        weight_coefficient = 1
    wgh_class = FairWeight(num_arms, weight_coefficient)
    weights = wgh_class.weights
    # The number of time steps (for higher discount factor should be set higher)
    num_steps = 100
    # The number of learning episodes
    num_episodes = 200
    # The data for multi-objective MDP and the dual form of it
    data_mrp = LPData(num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weights, discount)

    policy_flags = [0, 1]

    # ----------------------------------- DUAL MO-MDP Setup -----------------------------------
    if policy_flags[0] == 1:
        dlp_model = build_dlp(data_mrp)
        dlp_results, dlp_model = solve_dlp(model=dlp_model)
        extract_dlp(model=dlp_model, lp_data=data_mrp)
        dlp_csv_name = "results_dlp_ggi"
        env_dlp = MachineReplacement(
            num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, num_steps, dlp_csv_name
        )
        dlp_rewards = []

    # ----------------------------------- DQN Setup -----------------------------------
    if policy_flags[1] == 1:
        dqn_csv_name = "results_dqn_ggi"
        env_dqn = MachineReplacement(
            num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, num_steps, dqn_csv_name
        )
        # parser = ap.ArgumentParser('Hyper-parameters for DQNetwork')
        # parser.add_argument('l-rate', type=float, default=0.001, help='learning rate')
        # parser.add_argument('h-size', type=float, default=64, help='hidden layer size')
        # parser.add_argument('ep-max', type=float, default=1.0, help='initial epsilon')
        # parser.add_argument('ep-dec', type=float, default=0.99, help='decaying rate')
        # parser.add_argument('ep-min', type=float, default=0.01, help='ending epsilon')
        # args = parser.parse_args()
        agent1 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.001, 64)
        agent2 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.001, 128)
        agent3 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.001, 256)
        agent4 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.001, 512)
        agent5 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.01, 64)
        agent6 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.01, 128)
        agent7 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.01, 256)
        agent8 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.01, 512)
        agent9 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.1, 64)
        agent10 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.1, 128)
        agent11 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.1, 256)
        agent12 = ODQNAgent(data_mrp, discount, ggi_flag, weights, 0.1, 512)
        dqn1_rewards = []
        dqn2_rewards = []
        dqn3_rewards = []
        dqn4_rewards = []
        dqn5_rewards = []
        dqn6_rewards = []
        dqn7_rewards = []
        dqn8_rewards = []
        dqn9_rewards = []
        dqn10_rewards = []
        dqn11_rewards = []
        dqn12_rewards = []

    # ----------------------------------- Monte-Carlo Simulations -----------------------------------

    for i_episode in range(num_episodes):

        print("Episode: " + str(i_episode))

        if policy_flags[0] == 1:
            state = env_dlp.reset()
            dlp_reward = 0
            for t in range(num_steps):
                action = policy_dlp(state, dlp_model, data_mrp)
                next_observation, reward, done, _ = env_dlp.step(action)
                dlp_reward += discount ** t * reward
                if done:
                    break
                else:
                    state = np.zeros(num_arms)
                    for n in range(num_arms):
                        state[n] = int(next_observation[n] * num_states)
            rewards_sorted = np.sort(dlp_reward)
            dlp_rewards.append(np.dot(rewards_sorted, weights))

        if policy_flags[1] == 1:
            observation = env_dqn.reset()
            dqn1_reward = 0
            for t in range(num_steps):
                action = agent1.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn1_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent1.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn1_reward)
            dqn1_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn2_reward = 0
            for t in range(num_steps):
                action = agent2.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn2_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent2.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn2_reward)
            dqn2_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn3_reward = 0
            for t in range(num_steps):
                action = agent3.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn3_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent3.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn3_reward)
            dqn3_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn4_reward = 0
            for t in range(num_steps):
                action = agent4.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn4_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent4.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn4_reward)
            dqn4_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn5_reward = 0
            for t in range(num_steps):
                action = agent5.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn5_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent5.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn5_reward)
            dqn5_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn6_reward = 0
            for t in range(num_steps):
                action = agent6.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn6_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent6.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn6_reward)
            dqn6_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn7_reward = 0
            for t in range(num_steps):
                action = agent7.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn7_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent7.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn7_reward)
            dqn7_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn8_reward = 0
            for t in range(num_steps):
                action = agent8.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn8_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent8.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn8_reward)
            dqn8_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn9_reward = 0
            for t in range(num_steps):
                action = agent9.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn9_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent9.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn9_reward)
            dqn9_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn10_reward = 0
            for t in range(num_steps):
                action = agent10.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn10_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent10.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn10_reward)
            dqn10_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn11_reward = 0
            for t in range(num_steps):
                action = agent11.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn11_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent11.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn11_reward)
            dqn11_rewards.append(np.dot(rewards_sorted, weights))

            observation = env_dqn.reset()
            dqn12_reward = 0
            for t in range(num_steps):
                action = agent12.act(observation)
                next_observation, reward_list, done, _ = env_dqn.step(action)
                dqn12_reward += discount ** t * reward_list
                if done:
                    break
                else:
                    agent12.update(observation, action, reward_list, next_observation)
                    observation = next_observation
            rewards_sorted = np.sort(dqn12_reward)
            dqn12_rewards.append(np.dot(rewards_sorted, weights))

    if policy_flags[0] == 1:
        dlp_rewards = np.array(dlp_rewards)
        rewards_dlp = dlp_rewards.copy()
        for i in range(num_episodes):
            rewards_dlp[i] = np.mean(dlp_rewards[0:i])
    if policy_flags[1] == 1:
        dqn1_rewards = np.array(dqn1_rewards)
        rewards_dqn1 = dqn1_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn1[i] = np.mean(dqn1_rewards[0:i])
        dqn2_rewards = np.array(dqn2_rewards)
        rewards_dqn2 = dqn2_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn2[i] = np.mean(dqn2_rewards[0:i])
        dqn3_rewards = np.array(dqn3_rewards)
        rewards_dqn3 = dqn3_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn3[i] = np.mean(dqn3_rewards[0:i])
        dqn4_rewards = np.array(dqn4_rewards)
        rewards_dqn4 = dqn4_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn4[i] = np.mean(dqn4_rewards[0:i])
        dqn5_rewards = np.array(dqn5_rewards)
        rewards_dqn5 = dqn5_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn5[i] = np.mean(dqn5_rewards[0:i])
        dqn6_rewards = np.array(dqn6_rewards)
        rewards_dqn6 = dqn6_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn6[i] = np.mean(dqn6_rewards[0:i])
        dqn7_rewards = np.array(dqn7_rewards)
        rewards_dqn7 = dqn7_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn7[i] = np.mean(dqn7_rewards[0:i])
        dqn8_rewards = np.array(dqn8_rewards)
        rewards_dqn8 = dqn8_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn8[i] = np.mean(dqn8_rewards[0:i])
        dqn9_rewards = np.array(dqn8_rewards)
        rewards_dqn9 = dqn9_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn9[i] = np.mean(dqn9_rewards[0:i])
        dqn10_rewards = np.array(dqn10_rewards)
        rewards_dqn10 = dqn10_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn10[i] = np.mean(dqn10_rewards[0:i])
        dqn11_rewards = np.array(dqn11_rewards)
        rewards_dqn11 = dqn11_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn11[i] = np.mean(dqn11_rewards[0:i])
        dqn12_rewards = np.array(dqn12_rewards)
        rewards_dqn12 = dqn12_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn12[i] = np.mean(dqn12_rewards[0:i])

    # ----------------------------------------- Results -----------------------------------------

    # states_list = get_state_list(num_states, num_arms)
    # for state in states_list:
    #     observation = np.array(state) / num_states
    #     dqn_action = agent.act(observation)
    #     print("State: {} -> DQN Action: {}".format(str(state), str(dqn_action)))
    #     # mlp_action = policy_mlp(state, mlp_model, data_mrp)
    #     # dlp_action = policy_dlp(state, dlp_model, data_mrp)
    #     # print("State: {} -> DQN Action: {}, MLP Action: {}, DLP Action: {}"
    #     #       .format(str(state), str(dqn_action), str(mlp_action), str(dlp_action)))

    fig, ax = plt.subplots()
    if policy_flags[1] == 1:
        ax.plot(range(len(rewards_dqn1)), rewards_dqn1, label="DQN1")
        ax.plot(range(len(rewards_dqn2)), rewards_dqn2, label="DQN2")
        ax.plot(range(len(rewards_dqn3)), rewards_dqn3, label="DQN3")
        ax.plot(range(len(rewards_dqn4)), rewards_dqn4, label="DQN4")
        ax.plot(range(len(rewards_dqn5)), rewards_dqn5, label="DQN5")
        ax.plot(range(len(rewards_dqn6)), rewards_dqn6, label="DQN6")
        ax.plot(range(len(rewards_dqn7)), rewards_dqn7, label="DQN7")
        ax.plot(range(len(rewards_dqn8)), rewards_dqn8, label="DQN8")
        ax.plot(range(len(rewards_dqn9)), rewards_dqn9, label="DQN9")
        ax.plot(range(len(rewards_dqn10)), rewards_dqn10, label="DQN10")
        ax.plot(range(len(rewards_dqn11)), rewards_dqn11, label="DQN11")
        ax.plot(range(len(rewards_dqn12)), rewards_dqn12, label="DQN12")
    if policy_flags[0] == 1:
        ax.plot(range(len(rewards_dlp)), rewards_dlp, label="DLP")
    ax.set(xlabel='Episodes', ylabel='Discounted Reward',
           title='Learning Curve')
    ax.grid()
    plt.legend()
    plt.show()
