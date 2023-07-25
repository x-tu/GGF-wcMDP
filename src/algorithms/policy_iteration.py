"""This script is used to run policy iteration algorithm."""

import numpy as np

from src.env.mrp_env import MachineReplace
from utils.mrp import MRPData


class PIAgent:
    def __init__(self, env, gamma=0.99, theta=1e-10):
        self.env = env
        self.gamma = gamma  # discount factor
        self.theta = theta  # threshold to stop policy evaluation
        self.n_action = self.env.action_space.n
        self.n_state = self.env.observation_space.n

    def policy_evaluation(self, pi):
        prev_V = np.zeros(self.n_state, dtype=np.float64)
        while True:
            V = np.zeros(self.n_state, dtype=np.float64)
            for s in range(self.n_state):
                next_state, reward, done, _ = self.env.step(pi(s))
                # next_state_prob = self.env.mrp_data.bigT[s, :, pi(s)]
                V[s] += reward + self.gamma * prev_V[next_state] * (not done)
            if np.max(np.abs(prev_V - V)) < self.theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V):
        Q = np.zeros((self.n_state, self.n_action), dtype=np.float64)
        for s in range(self.n_state):
            for a in range(self.n_action):
                next_state, reward, done, info = self.env.step(a)
                Q[s][a] += reward + self.gamma * V[next_state] * (not done)
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi

    def run_policy_iteration(self):
        # random_actions = [random.choice(range(n_action)) for s in range(self.env.observation_space.n)]
        random_actions = [0 for s in range(self.env.observation_space.n)]
        pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
        while True:
            old_pi = {s: pi(s) for s in range(self.env.observation_space.n)}
            V = self.policy_evaluation(pi)
            pi = self.policy_improvement(V)
            if old_pi == {s: pi(s) for s in range(self.env.observation_space.n)}:
                break
        return V, pi


# initialize the environment
env = MachineReplace(
    n_group=3, n_state=3, n_action=2, init_state=0, out_csv_name="QL.csv", ggi=False
)

mrp_data = MRPData(n_group=3, n_state=3, n_action=2)
# initialize the agent
policy_agent = PIAgent(env, gamma=0.9, theta=1e-10)
# run policy iteration
V, pi = policy_agent.run_policy_iteration()

print("V: ", V)
print("pi: ", {s: pi(s) for s in range(env.observation_space.n)})
