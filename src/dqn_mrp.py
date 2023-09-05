import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class ODQNAgent:
    def __init__(self,
                 data_mrp,
                 discount,
                 ggi_flag,
                 weights,
                 l_rate=1e-3,
                 h_size=64,
                 epsilon=1.0,
                 epsilon_decay=0.99,
                 min_epsilon=0.01):
        self.data_mrp = data_mrp
        self.ggi_flag = ggi_flag
        input_dim = 2 * data_mrp.num_arms
        output_dim = 1
        if ggi_flag:
            output_dim = data_mrp.num_arms
        self.q_network = DQNetwork(input_dim, h_size, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=l_rate)
        self.weights = weights
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.loss_fn = nn.MSELoss()

    def act(self, observation):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.data_mrp.num_actions)
        # Exploitation
        else:
            # The Q-values for GGF case
            qg_values = torch.zeros(self.data_mrp.num_actions, self.data_mrp.num_arms)
            q_ggfvalues = torch.zeros(self.data_mrp.num_actions)
            # The Q-values for non-GGF case
            q_values = torch.zeros(self.data_mrp.num_actions)
            # Loop over all actions
            for a_idx in range(self.data_mrp.num_actions):
                # The one-hot-encoded action
                temp_action = self.data_mrp.action_tuples[a_idx]
                # The input to the DQN
                nn_input = np.hstack((observation, np.array(temp_action)))
                if self.ggi_flag:
                    # The exploitation action is selected according to argmax_{a} GGF(q(s, a)) or argmax_{a} GGF(r(s, a) + gamma q(s, a))?
                    # Get the per-step reward
                    state_list = observation * self.data_mrp.num_states
                    action_list = np.zeros(self.data_mrp.num_arms, dtype=int)
                    if a_idx > 0:
                        action_list[a_idx - 1] = 1
                    reward_list = np.zeros(self.data_mrp.num_arms)
                    for n in range(self.data_mrp.num_arms):
                        reward_list[n] = self.data_mrp.rewards[int(state_list[n]), n, action_list[n]]
                    qg_values[a_idx, :] = self.q_network(torch.tensor(nn_input).float())
                    q_values_sorted = torch.sort(torch.tensor(reward_list) + self.discount * qg_values[a_idx, :])
                    for w in range(len(self.weights)):
                        q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * q_values_sorted[0][w]
                else:
                    # The exploitation action is selected according to argmax_{a} q(s, a)
                    q_values[a_idx] = self.q_network(torch.tensor(nn_input).float())
            if self.ggi_flag:
                return torch.argmax(q_ggfvalues).item()
            else:
                return torch.argmax(q_values).item()

    def update(self, observation, action, reward, next_observation):
        # Required variables for GGI case
        qg_values = torch.zeros(self.data_mrp.num_actions, self.data_mrp.num_arms)
        next_qg_values = torch.zeros(self.data_mrp.num_actions, self.data_mrp.num_arms)
        rq_ggfvalues = torch.zeros(self.data_mrp.num_actions)
        next_rq_ggfvalues = torch.zeros(self.data_mrp.num_actions)
        # Required variables for non-GGI case
        q_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
        next_q_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
        # Loop over all actions
        for a_idx in range(self.data_mrp.num_actions):
            # The one-hot-encoded action
            temp_action = self.data_mrp.action_tuples[a_idx]
            # The current_state input to the DQN
            nn_input = np.hstack((observation, np.array(temp_action)))
            # The next_state input to the DQN
            next_nn_input = np.hstack((next_observation, np.array(temp_action)))
            # For GGF case
            if self.ggi_flag:
                # Get the q_values for the current and next input
                qg_values[a_idx, :] = self.q_network(torch.tensor(nn_input).float())
                next_qg_values[a_idx, :] = self.q_network(torch.tensor(next_nn_input).float())
                # Get the per-step reward
                state_list = observation * self.data_mrp.num_states
                action_list = np.zeros(self.data_mrp.num_arms, dtype=int)
                if a_idx > 0:
                    action_list[a_idx - 1] = 1
                reward_list = np.zeros(self.data_mrp.num_arms)
                for n in range(self.data_mrp.num_arms):
                    reward_list[n] = self.data_mrp.rewards[int(state_list[n]), n, action_list[n]]
                # Get the GGF(r(s, a) + discount * q(s', a))
                nextrq_ggfvalues_sorted = torch.sort(torch.tensor(reward_list) + self.discount * next_qg_values[a_idx, :])
                for w in range(len(self.weights)):
                    next_rq_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * nextrq_ggfvalues_sorted[0][w]
            # For non-GGF case
            else:
                q_values[a_idx] = self.q_network(torch.tensor(nn_input).float())
                next_q_values[a_idx] = self.q_network(torch.tensor(next_nn_input).float())
        if self.ggi_flag:
            # Compute the target: Loss between q(s, a) and r(s, a) + discount * q(s', a_max) where
            # a_max = argmax_{a} GGF(r(s, a) + discount * q(s', a)) is the greedy action for the next_state.
            next_greedy_action = torch.argmax(next_rq_ggfvalues).item()
            target_ggf = torch.tensor(reward) + self.discount * next_qg_values[next_greedy_action]
            target_qg_values = qg_values.clone()
            target_qg_values[action] = target_ggf
            # Compute the loss
            loss = self.loss_fn(qg_values, target_qg_values)
        else:
            # Compute the target: average reward over arms + discount * q(s', a_max) where
            # a_max = argmax_{a} q(s', a) is the greedy action for the next_state.
            next_greedy_action = torch.argmax(next_q_values).item()
            target = np.dot(reward, self.weights) + self.discount * next_q_values[next_greedy_action].item()
            target_q_values = q_values.clone()
            target_q_values[action] = target
            # Compute the loss
            loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


class RDQNAgent:
    def __init__(self,
                 data_mrp,
                 discount,
                 ggi_flag,
                 weights,
                 l_rate=1e-3,
                 h_size=64,
                 epsilon=1.0,
                 epsilon_decay=0.99,
                 min_epsilon=0.01):
        self.data_mrp = data_mrp
        self.ggi_flag = ggi_flag
        input_dim = data_mrp.num_arms
        output_dim = data_mrp.num_actions
        if ggi_flag:
            output_dim = data_mrp.num_arms * data_mrp.num_actions
        self.q_network = DQNetwork(input_dim, h_size, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=l_rate)
        self.weights = weights
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.loss_fn = nn.MSELoss()

    def act(self, observation, prev_reward_list):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.data_mrp.num_actions)
        # Exploitation
        else:
            # The Q-values
            q_values = self.q_network(torch.tensor(observation).float())
            if self.ggi_flag:
                # The Q-values for GGF case
                q_ggfvalues = torch.zeros(self.data_mrp.num_actions)
                # Loop over all actions
                for a_idx in range(self.data_mrp.num_actions):
                    # The exploitation action is selected according to argmax_{a} GGF(r(s, a) + discount * q(s, a))
                    temp_var = torch.tensor(prev_reward_list, dtype=torch.float32) + self.discount * q_values[a_idx * self.data_mrp.num_arms:(a_idx+1) * self.data_mrp.num_arms]
                    q_values_sorted = torch.sort(temp_var)
                    for w in range(len(self.weights)):
                        q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * q_values_sorted[0][w]
                return torch.argmax(q_ggfvalues).item()
            else:
                return torch.argmax(q_values).item()

    def update(self, observation, action, reward, next_observation):
        # The Q-values
        q_values = self.q_network(torch.tensor(observation).float())
        next_q_values = self.q_network(torch.tensor(next_observation).float())
        # For GGF case
        if self.ggi_flag:
            # Required variables for GGI case
            q_ggfvalues = torch.zeros(self.data_mrp.num_actions)
            next_q_ggfvalues = torch.zeros(self.data_mrp.num_actions)
            # Loop over all actions
            for a_idx in range(self.data_mrp.num_actions):
                temp_var = q_values[a_idx * self.data_mrp.num_arms:(a_idx+1) * self.data_mrp.num_arms]
                next_temp_var = next_q_values[a_idx * self.data_mrp.num_arms:(a_idx+1) * self.data_mrp.num_arms]
                q_ggfvalues_sorted = torch.sort(temp_var)
                nextq_ggfvalues_sorted = torch.sort(next_temp_var)
                for w in range(len(self.weights)):
                    q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * q_ggfvalues_sorted[0][w]
                    next_q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * nextq_ggfvalues_sorted[0][w]
            next_greedy_action = torch.argmax(next_q_ggfvalues).item()
            target_ggfsorted = torch.sort(torch.tensor(reward) + self.discount * next_q_ggfvalues[next_greedy_action])
            target_ggf = 0
            for w in range(len(self.weights)):
                target_ggf += torch.tensor(self.weights)[w] * target_ggfsorted[0][w]
            target_q_ggfvalues = q_ggfvalues.clone()
            target_q_ggfvalues[action] = target_ggf
            # Compute the loss
            loss = self.loss_fn(q_ggfvalues, target_q_ggfvalues)
        else:
            # Compute the target: average reward over arms + discount * q(s', a_max) where
            # a_max = argmax_{a} q(s', a) is the greedy action for the next_state.
            next_greedy_action = torch.argmax(next_q_values).item()
            target = np.dot(reward, self.weights) + self.discount * next_q_values[next_greedy_action].item()
            target_q_values = q_values.clone()
            target_q_values[action] = target
            # Compute the loss
            loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
