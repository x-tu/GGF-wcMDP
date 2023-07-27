import numpy as np


class PIAgent:
    def __init__(self, data, discount, epsilon=1e-10):
        self.data = data
        self.discount = discount  # discount factor
        self.epsilon = epsilon  # threshold to stop policy evaluation
        self.num_states = len(self.data.state_indices)
        self.num_actions = len(self.data.action_indices)

    def policy_evaluation(self, policy):
        old_values = np.zeros(self.num_states, dtype=np.float64)
        while True:
            new_values = np.zeros(self.num_states, dtype=np.float64)
            for s in range(self.num_states):
                next_state_prob = self.data.global_transitions[s, :, policy(s)]
                reward = self.data.global_costs[s, policy(s), :]
                new_values[s] += reward + self.gamma * np.dot(old_values, next_state_prob)
            if np.max(np.abs(old_values - new_values)) < self.theta:
                break
            old_values = new_values.copy()
        return new_values

    def policy_improvement(self, v_values):
        q_values = np.zeros((self.num_states, self.n_action), dtype=np.float64)
        for s in range(self.num_states):
            for a in range(self.n_action):
                next_state_prob = self.data.global_transitions[s, :, a]
                reward = sum(-self.data.global_costs[s, a, :] + 110)
                q_values[s][a] += reward + self.gamma * np.dot(v_values, next_state_prob)
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(q_values, axis=1))}[s]
        return new_pi

    def run_policy_iteration(self):
        init_actions = [0 for s in range(self.num_states)]
        pi = lambda s: {s: 0 for s, a in enumerate(init_actions)}[s]
        while True:
            old_pi = {s: pi(s) for s in range(self.num_states)}
            if not self.ggi:
                Vs = self.policy_evaluation(pi)
                pi = self.policy_improvement(Vs)
            else:
                # TODO: add GGF policy evaluation and improvement
                Vs = self.policy_evaluation(pi)
                pi = self.policy_improvement(Vs)
            # convergence check
            if old_pi == {s: pi(s) for s in range(self.num_states)}:
                break
        return Vs, pi

