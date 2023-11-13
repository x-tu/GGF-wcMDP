import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACNetwork(nn.Module):
    """Deep Q Network with 3 fully connected layers with ReLu activation function."""

    def __init__(
        self,
        input_dim: int,
        output_dim_policy: int,
        output_dim_value: int,
        hidden_dims: tuple = (32, 32),
        activation_fc: nn.Module = F.relu,
    ):
        super(ACNetwork, self).__init__()
        self.activation_fc = activation_fc
        # define the network
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        # define the output layers
        self.value_output_layer = nn.Linear(hidden_dims[-1], output_dim_value)
        self.policy_output_layer = nn.Linear(hidden_dims[-1], output_dim_policy)

    def format(self, state):
        """Used to format the input state to the network."""

        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        """Forward pass of the network. Returns the logits for the policy and the value estimate."""

        x = self.format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.policy_output_layer(x), self.value_output_layer(x)

    def full_pass(self, state: np.array):
        """Full pass of the network.

        Args:
            state (np.array): The state of the environment.
        """

        # define the logits for the policy and the value estimate
        logits, values = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        # define the log probability of the action
        log_prob = dist.log_prob(action).unsqueeze(-1)
        # define the entropy which is used to encourage exploration
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        # define whether the action is exploratory
        is_exploratory = action != np.argmax(
            logits.detach().numpy(), axis=int(len(state) != 1)
        )
        return action, is_exploratory, log_prob, entropy, values

    def select_action(self, state):
        """Select an action based on the policy."""

        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action

    def select_greedy_action(self, state):
        """Select the greedy action based on the policy."""

        logits, _ = self.forward(state)
        return np.argmax(logits.detach().numpy())

    def evaluate_state(self, state):
        _, value = self.forward(state)
        return value
