"""This script is used to run policy iteration algorithm."""

import numpy as np

from utils.mrp_lp import MRPData


class PIAgent:
    def __init__(self, params, gamma=0.99, theta=1e-10):
        self.mrp_data = MRPData(
            n_group=params["n_group"],
            n_state=params["n_state"],
            n_action=params["n_action"],
            weight=None,
        )
        self.ggi = params["ggi"]
        self.gamma = gamma  # discount factor
        self.theta = theta  # threshold to stop policy evaluation
        self.n_state = len(self.mrp_data.idx_list_s)
        self.n_action = len(self.mrp_data.idx_list_a)
        self.n_group = self.mrp_data.n_group

    def policy_evaluation(self, pi):
        if not self.ggi:
            prev_V = np.zeros(self.n_state, dtype=np.float64)
            while True:
                V = np.zeros(self.n_state, dtype=np.float64)
                for s in range(self.n_state):
                    next_state_prob = self.mrp_data.bigT[s, :, pi(s)]
                    reward = sum(-self.mrp_data.bigC[s, pi(s), :] + 110)
                    V[s] += reward + self.gamma * np.dot(prev_V, next_state_prob)
                if np.max(np.abs(prev_V - V)) < self.theta:
                    break
                prev_V = V.copy()
        else:
            prev_V = np.zeros((self.n_state, self.n_group), dtype=np.float64)
            while True:
                V = np.zeros((self.n_state, self.n_group), dtype=np.float64)
                for s in range(self.n_state):
                    next_state_prob = self.mrp_data.bigT[s, :, pi(s)]
                    reward = -self.mrp_data.bigC[s, pi(s), :] + 110
                    for g in range(self.n_group):
                        V[s, g] += reward[g] + self.gamma * np.dot(
                            prev_V[:, g], next_state_prob
                        )
                if np.max(np.abs(prev_V - V)) < self.theta:
                    break
                prev_V = V.copy()
        return V

    def policy_improvement(self, V):
        Q = np.zeros((self.n_state, self.n_action), dtype=np.float64)
        if not self.ggi:
            for s in range(self.n_state):
                for a in range(self.n_action):
                    next_state_prob = self.mrp_data.bigT[s, :, a]
                    reward = sum(-self.mrp_data.bigC[s, a, :] + 110)
                    Q[s][a] += reward + self.gamma * np.dot(V, next_state_prob)
            new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        else:
            Qn = np.zeros(self.n_group, dtype=np.float64)
            for s in range(self.n_state):
                for a in range(self.n_action):
                    next_state_prob = self.mrp_data.bigT[s, :, a]
                    reward = -self.mrp_data.bigC[s, a, :] + 110
                    for g in range(self.n_group):
                        Qn[g] += reward[g] + self.gamma * np.dot(
                            V[:, g], next_state_prob
                        )
                    Q[s][a] = np.dot([1, 0.5, 0.25], np.sort(Qn))
            new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi

    def run_policy_iteration(self):
        # set initial actions to 0
        init_actions = [0 for s in range(self.n_state)]
        pi = lambda s: {s: 0 for s, a in enumerate(init_actions)}[s]
        while True:
            old_pi = {s: pi(s) for s in range(self.n_state)}
            # TODO: add GGF policy evaluation and improvement
            Vs = self.policy_evaluation(pi)
            pi = self.policy_improvement(Vs)
            # convergence check
            if old_pi == {s: pi(s) for s in range(self.n_state)}:
                break
        return Vs, pi


# initialize the environment
params = {"n_group": 3, "n_state": 3, "n_action": 2, "ggi": True}
# initialize the agent
policy_agent = PIAgent(params, gamma=0.99, theta=1e-10)
# run policy iteration
V, pi = policy_agent.run_policy_iteration()
policy = {s: pi(s) for s in range(policy_agent.n_state)}

# results for fixing the RL policy
from solver.fix_policy import (
    build_ggf_fix,
    extract_results as extract_results_fix,
    solve_ggf_fix,
)

model = build_ggf_fix(policy_agent.mrp_data, policy)
# Solve the GGF model
results, ggf_model = solve_ggf_fix(model=model)
extract_results_fix(model=ggf_model, data=policy_agent.mrp_data, policy_rl=policy)
