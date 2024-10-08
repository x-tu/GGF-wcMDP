"""This script is used to run policy iteration algorithm (GGF and non-GGF)."""

import numpy as np

# results for fixing the RL policy
from solver.fix_policy import (
    build_ggf_fix,
    extract_results as extract_results_fix,
    solve_ggf_fix,
)
from utils.mrp import MRPData


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
        self.q_table = np.zeros((self.n_state, self.n_action), dtype=np.float64)

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
            for s in range(self.n_state):
                for a in range(self.n_action):
                    next_state_prob = self.mrp_data.bigT[s, :, a]
                    reward = -self.mrp_data.bigC[s, a, :] + 110
                    Qn = reward + self.gamma * np.dot(V.T, next_state_prob)
                    Q[s][a] += np.dot([1, 0.5, 0.25], np.sort(Qn))
            new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        self.q_table = Q
        return new_pi

    def run_policy_iteration(self):
        # set initial actions to 0
        init_actions = [0 for s in range(self.n_state)]
        pi = lambda s: {s: 0 for s, a in enumerate(init_actions)}[s]
        iteration = 0
        rewards = []
        while True:
            old_pi = {s: pi(s) for s in range(self.n_state)}
            # TODO: add GGF policy evaluation and improvement
            Vs = self.policy_evaluation(pi)
            pi = self.policy_improvement(Vs)
            iteration += 1
            # convergence check
            new_pi = {s: pi(s) for s in range(self.n_state)}
            if old_pi == new_pi:
                break
            model = build_ggf_fix(self.mrp_data, new_pi)
            # Solve the GGF model
            results, ggf_model = solve_ggf_fix(model=model)
            reward = extract_results_fix(
                model=ggf_model, data=self.mrp_data, policy_rl=new_pi
            )
            rewards.append(reward)
        return Vs, pi
