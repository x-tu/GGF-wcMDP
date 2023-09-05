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

    # TODO: Save figures
    # TODO: Model generalization

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
    ggi_flag = True
    # The fair weight coefficient
    if ggi_flag:
        weight_coefficient = 2
    else:
        weight_coefficient = 1
    wgh_class = FairWeight(num_arms, weight_coefficient)
    weights = wgh_class.weights
    # The number of time steps (for higher discount factor should be set higher)
    num_steps = 100
    # The number of learning episodes
    num_episodes = 1000
    # The data for multi-objective MDP and the dual form of it
    data_mrp = LPData(num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weights, discount)

    policy_flags = [1, 1]

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
        # agent1 = ODQNAgent(data_mrp, discount, ggi_flag, weights, l_rate=1e-3, h_size=128)
        # dqn1_rewards = []
        agent2 = RDQNAgent(data_mrp, discount, ggi_flag, weights, l_rate=1e-3, h_size=128)
        dqn2_rewards = []

    # ----------------------------------- Monte-Carlo Simulations -----------------------------------

    for i_episode in range(num_episodes):

        print("Episode: " + str(i_episode))

        if policy_flags[0] == 1:
            dlp_epochrewards = []
            for ep in range(10):
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
                dlp_epochrewards.append(dlp_reward)
            rewards_sorted = np.sort(sum(dlp_epochrewards)/len(dlp_epochrewards))
            dlp_rewards.append(np.dot(rewards_sorted, weights))

        if policy_flags[1] == 1:
            dqn2_epochrewards = []
            # dqn1_epochrewards = []
            for ep in range(10):
                # observation = env_dqn.reset()
                # dqn1_reward = 0
                # for t in range(num_steps):
                #     action = agent1.act(observation)
                #     next_observation, reward_list, done, _ = env_dqn.step(action)
                #     dqn1_reward += discount ** t * reward_list
                #     if done:
                #         break
                #     else:
                #         agent1.update(observation, action, reward_list, next_observation)
                #         observation = next_observation
                # dqn1_epochrewards.append(dqn1_reward)

                observation = env_dqn.reset()
                reward_list = [0] * num_arms
                dqn2_reward = 0
                for t in range(num_steps):
                    action = agent2.act(observation, reward_list)
                    next_observation, reward_list, done, _ = env_dqn.step(action)
                    dqn2_reward += discount ** t * reward_list
                    if done:
                        break
                    else:
                        agent2.update(observation, action, reward_list, next_observation)
                        observation = next_observation
                    dqn2_epochrewards.append(dqn2_reward)

            # rewards_sorted = np.sort([sum(col) / len(col) for col in zip(*dqn1_epochrewards)])
            # dqn1_rewards.append(np.dot(rewards_sorted, weights))

            rewards_sorted = np.sort([sum(col) / len(col) for col in zip(*dqn2_epochrewards)])
            dqn2_rewards.append(np.dot(rewards_sorted, weights))

    if policy_flags[0] == 1:
        dlp_rewards = np.array(dlp_rewards)
        rewards_dlp = dlp_rewards.copy()
        for i in range(num_episodes):
            rewards_dlp[i] = np.mean(dlp_rewards[i-1:i])
    if policy_flags[1] == 1:
        # dqn1_rewards = np.array(dqn1_rewards)
        # rewards_dqn1 = dqn1_rewards.copy()
        # for i in range(num_episodes):
        #     rewards_dqn1[i] = np.mean(dqn1_rewards[i-1:i])
        dqn2_rewards = np.array(dqn2_rewards)
        rewards_dqn2 = dqn2_rewards.copy()
        for i in range(num_episodes):
            rewards_dqn2[i] = np.mean(dqn2_rewards[i-1:i])

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
        # ax.plot(range(len(rewards_dqn1)), rewards_dqn1, label="DQN1")
        ax.plot(range(len(rewards_dqn2)), rewards_dqn2, label="DQN")
    if policy_flags[0] == 1:
        ax.plot(range(len(rewards_dlp)), rewards_dlp, label="DLP")
    ax.set(xlabel='Episodes', ylabel='Discounted Reward',
           title='Learning Curve')
    ax.grid()
    plt.legend()
    plt.show()
    plt.savefig('learning.png')
