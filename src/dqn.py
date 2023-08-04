import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self, observation_dim, num_arms, ggi_flag):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_dim + num_arms, 64)
        self.fc2 = nn.Linear(64, 64)
        if ggi_flag:
            self.fc3 = nn.Linear(64, num_arms)
        else:
            self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self,
                 data_mrp,
                 ggi_flag,
                 weights,
                 discount=0.99,
                 initial_lr=1e-3,
                 epsilon=1.0,
                 epsilon_decay=0.99,
                 min_epsilon=0.01):
        self.data_mrp = data_mrp
        self.ggi_flag = ggi_flag
        self.q_network = DQNetwork(data_mrp.num_states, data_mrp.num_arms, ggi_flag)
        self.weights = weights
        self.discount = discount
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=initial_lr)
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
            q_ggfvalues = torch.tensor(np.zeros(self.data_mrp.num_actions))
            # The Q-values for non-GGF case
            q_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
            # Loop over all actions
            for a_idx in range(self.data_mrp.num_actions):
                # The one-hot-encoded action
                temp_action = self.data_mrp.action_tuples[a_idx]
                # The input to the DQN
                nn_input = np.hstack((observation, np.array(temp_action)))
                if self.ggi_flag:
                    # The exploitation action is selected according to argmax_{a} GGF(q(s, a))
                    q_values_sorted = torch.sort(self.q_network(torch.tensor(nn_input).float()))
                    for w in range(len(self.weights)):
                        q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * q_values_sorted[0][w]
                    return torch.argmax(q_ggfvalues).item()
                else:
                    # The exploitation action is selected according to argmax_{a} q(s, a)
                    q_values[a_idx] = self.q_network(torch.tensor(nn_input).float())
                    return torch.argmax(q_values).item()

    def update(self, observation, action, reward, next_observation, done):
        # Required variables for GGI case
        q_ggfvalues = torch.tensor(np.zeros(self.data_mrp.num_actions))
        next_q_ggfvalues = torch.tensor(np.zeros(self.data_mrp.num_actions))
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
            if self.ggi_flag:
                q_ggfvalues_sorted = torch.sort(self.q_network(torch.tensor(nn_input).float()))
                nextq_ggfvalues_sorted = torch.sort(self.q_network(torch.tensor(next_nn_input).float()))
                for w in range(len(self.weights)):
                    q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * q_ggfvalues_sorted[0][w]
                    next_q_ggfvalues[a_idx] += torch.tensor(self.weights)[w] * nextq_ggfvalues_sorted[0][w]
            else:
                q_values[a_idx] = self.q_network(torch.tensor(nn_input).float())
                next_q_values[a_idx] = self.q_network(torch.tensor(next_nn_input).float())
        if self.ggi_flag:
            # The greedy action for the next_state
            next_greedy_action = torch.argmax(next_q_ggfvalues).item()
            # Compute the target
            target_ggfsorted = torch.sort(torch.tensor(reward) + self.discount * next_q_ggfvalues[next_greedy_action] * (1 - done))
            target_ggf = 0
            for w in range(len(self.weights)):
                target_ggf += torch.tensor(self.weights)[w] * target_ggfsorted[0][w]
            target_q_ggfvalues = q_ggfvalues.clone()
            target_q_ggfvalues[action] = target_ggf
            # Compute the loss
            loss = self.loss_fn(q_ggfvalues, target_q_ggfvalues)
        else:
            # The greedy action for the next_state
            next_greedy_action = torch.argmax(next_q_values).item()
            # Compute the target
            target = np.mean(reward) + self.discount * next_q_values[next_greedy_action] * (1 - done)
            target_q_values = q_values.clone()
            target_q_values[action] = target
            # Compute the loss
            loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
