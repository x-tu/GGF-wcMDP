import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, observation_dim, num_arms, ggi_flag):
        super(DQNNetwork, self).__init__()
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
        self.q_network = DQNNetwork(data_mrp.num_states, data_mrp.num_arms, ggi_flag)
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
            q_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
            for a_idx in range(self.data_mrp.num_actions):
                temp_action = self.data_mrp.action_tuples[a_idx]
                nn_input = np.hstack((observation, np.array(temp_action)))
                if self.ggi_flag:
                    state_list = observation * self.data_mrp.num_states
                    action_list = np.zeros(self.data_mrp.num_arms, dtype=int)
                    if a_idx > 0:
                        action_list[a_idx - 1] = 1
                    reward_list = np.zeros(self.data_mrp.num_arms)
                    for n in range(self.data_mrp.num_arms):
                        reward_list[n] = 1 - self.data_mrp.costs[int(state_list[n]), n, action_list[n]]
                    q_values_sorted = torch.sort(torch.tensor(reward_list) + self.discount * self.q_network(torch.tensor(nn_input).float()))
                    # q_values = torch.dot(q_values_sorted, torch.tensor(self.weights))
                    for w in range(len(self.weights)):
                        q_values[a_idx] += torch.tensor(self.weights)[w] * q_values_sorted[0][w]
                else:
                    q_values[a_idx] = self.q_network(torch.tensor(nn_input).float())
            return torch.argmax(q_values).item()

    def update(self, observation, action, reward, next_observation, done):
        rq_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
        next_rq_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
        q_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
        next_q_values = torch.tensor(np.zeros(self.data_mrp.num_actions))
        for a_idx in range(self.data_mrp.num_actions):
            temp_action = self.data_mrp.action_tuples[a_idx]
            nn_input = np.hstack((observation, np.array(temp_action)))
            next_nn_input = np.hstack((next_observation, np.array(temp_action)))
            if self.ggi_flag:
                rq_values_sorted = torch.sort(torch.tensor(reward) + self.discount * self.q_network(torch.tensor(nn_input).float()) * (1 - done))
                next_rq_values_sorted = torch.sort(torch.tensor(reward) + self.discount * self.q_network(torch.tensor(next_nn_input).float()) * (1 - done))
                # rq_values = torch.dot(rq_values_sorted, torch.tensor(self.weights))
                # next_rq_values = torch.dot(next_rq_values_sorted, torch.tensor(self.weights))
                for w in range(len(self.weights)):
                    rq_values[a_idx] += torch.tensor(self.weights)[w] * rq_values_sorted[0][w]
                    next_rq_values[a_idx] += torch.tensor(self.weights)[w] * next_rq_values_sorted[0][w]
            else:
                q_values[a_idx] = self.q_network(torch.tensor(nn_input).float())
                next_q_values[a_idx] = self.q_network(torch.tensor(next_nn_input).float())
        if self.ggi_flag:
            target = torch.max(next_rq_values).item()
            target_rq_values = rq_values.clone()
            target_rq_values[action] = target
            loss = self.loss_fn(rq_values, target_rq_values)
        else:
            target = np.mean(reward) + self.discount * torch.max(next_q_values).item() * (1 - done)
            target_q_values = q_values.clone()
            target_q_values[action] = target
            loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
