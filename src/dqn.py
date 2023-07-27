import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self,
                 observation_dim,
                 action_dim,
                 discount=0.99,
                 initial_lr=1e-3,
                 epsilon=1.0,
                 epsilon_decay=0.99,
                 min_epsilon=0.01):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_network = DQNNetwork(observation_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=initial_lr)
        self.loss_fn = nn.MSELoss()

    def act(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.q_network(torch.tensor(observation).float())
        return torch.argmax(q_values).item()

    def update(self, observation, action, reward, next_observation, done):
        q_values = self.q_network(torch.tensor(observation).float())
        next_q_values = self.q_network(torch.tensor(next_observation).float())

        target = reward + self.discount * torch.max(next_q_values).item() * (1 - done)
        target_q_values = q_values.clone()
        target_q_values[action] = target

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
