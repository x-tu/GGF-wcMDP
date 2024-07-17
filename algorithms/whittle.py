"""Code by Nima Akbarzadeh for Risk-Neutral Whittle Index Policy for Restless Bandit Problems"""

import numpy as np


class Whittle:
    def __init__(
        self,
        num_states: int,
        num_arms: int,
        reward,
        transition,
        horizon,
        discount_rate=0.95,
    ):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward  # (arm, x, act)
        self.transition = transition  # (arm, x, x', a)
        self.discount_rate = discount_rate
        self.horizon = horizon
        self.digits = 4
        self.w_indices = []

    def is_equal_mat(self, mat1, mat2):
        for t in range(self.horizon):
            if np.array_equal(mat1[:, t], mat2[:, t]) == False:
                return False
        return True

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            if np.any((ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)):
                print("Not indexable!")
                return False, np.zeros((self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0))
                for e in elements:
                    arm_indices[e, t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            upb_pol, _, _ = self.backward(arm, upper_bound)
            # ref_pol, _ = self.value_iteration(arm, penalty_ref)
            # upb_pol, _ = self.value_iteration(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                nxt_pol, _, _ = self.backward(arm, penalty)
                # nxt_pol, _ = self.value_iteration(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol):
                    flag, arm_indices = self.indexability_check(
                        arm_indices, nxt_pol, ref_pol, penalty
                    )
                    break
                else:
                    if self.is_equal_mat(nxt_pol, ref_pol) == False:
                        flag, arm_indices = self.indexability_check(
                            arm_indices, nxt_pol, ref_pol, penalty
                        )
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def backward(self, arm, penalty):
        # Value function initialization
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)
        # State-action value function
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)
        # Policy function
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)
        # Backward induction timing
        t = self.horizon - 1
        # The value iteration loop
        while t >= 0:
            # Loop over the first dimension of the state space
            for x in range(self.num_x):
                # Get the state-action value functions
                for act in range(2):
                    Q[x, t, act] = (
                        self.reward[0, x, act]
                        - penalty * act
                        + self.discount_rate
                        * np.dot(V[:, t + 1], self.transition[arm, x, :, act])
                    )
                # Get the value function and the policy
                if Q[x, t, 1] < Q[x, t, 0]:
                    V[x, t] = Q[x, t, 0]
                    pi[x, t] = 0
                else:
                    V[x, t] = Q[x, t, 1]
                    pi[x, t] = 1
            t = t - 1
        return pi, V, Q

    def value_iteration(self, arm, penalty):
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)

        for t in range(self.horizon - 1, -1, -1):
            for x in range(self.num_x):
                Q = np.zeros(2)
                for act in range(2):
                    Q[act] = self.reward[arm, x, act] - penalty * act
                    for next_x in range(self.num_x):
                        Q[act] += self.transition[arm, x, next_x, act] * (
                            self.discount_rate * V[next_x, t + 1]
                        )

                best_action = np.argmax(Q)
                V[x, t] = Q[best_action]
                pi[x, t] = best_action
        return pi, V

    @staticmethod
    def Whittle_policy(
        whittle_indices,
        n_selection,
        current_x,
        current_t,
        shuffle_indices=True,
        force_to_use_all_resources=False,
    ):
        num_a = len(whittle_indices)
        current_indices = np.array(
            [whittle_indices[arm][current_x[arm], current_t] for arm in range(num_a)]
        )
        if shuffle_indices:
            # Sort indices based on values and shuffle indices with same values
            sorted_indices = np.argsort(current_indices)[::-1]
            unique_indices, counts = np.unique(
                current_indices[sorted_indices], return_counts=True
            )
            top_indices = []
            top_len = 0
            for idx in range(len(unique_indices)):
                indices = np.where(
                    current_indices == unique_indices[len(unique_indices) - idx - 1]
                )[0]
                shuffled_indices = np.random.permutation(indices)
                # sort the indices from low to high
                if top_len + len(shuffled_indices) < n_selection:
                    top_indices.extend(list(shuffled_indices))
                    top_len += len(shuffled_indices)
                elif top_len + len(shuffled_indices) == n_selection:
                    top_indices.extend(list(shuffled_indices))
                    top_len += len(shuffled_indices)
                    break
                else:
                    top_indices.extend(list(shuffled_indices[: n_selection - top_len]))
                    top_len += len(shuffled_indices[: n_selection - top_len])
                    break
        else:
            top_indices = np.argsort(-current_indices)[:n_selection]
        # Create action vector
        action_vector = np.zeros_like(current_indices, dtype=np.int32)
        action_vector[top_indices] = 1
        if not force_to_use_all_resources:
            # if the whittle index values are negative, do not select any action
            for arm in range(num_a):
                if current_indices[arm] < 0:
                    action_vector[arm] = 0
        return action_vector
