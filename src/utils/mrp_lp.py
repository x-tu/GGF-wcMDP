import itertools
import random

import numpy as np

# random.seed(52)


class MRPData:
    def __init__(self, n_group=2, n_state=3, n_action=2, weight=None):
        self.n_group = n_group
        self.n_state = n_state
        self.n_action = n_action
        # TODO: validate the weights sum to 1
        if weight is None:
            weight = [1 / n_group] * n_group
        # normalize the weight
        elif sum(weight) != 1:
            sum_weight = sum(weight)
            weight = [w / sum_weight for w in weight]
        elif len(weight) != n_group:
            weight = [1 / n_group] * n_group
        self.weight = weight
        # TODO: generalization
        self.operation_cost = [10, 20, 50]
        self.replace_cost = 100
        self.discount = 0.9
        self.transition_matrix = np.array(
            [
                [[0.5, 0.5], [0.5, 0.5], [0, 0]],
                [[0, 0.5], [0.5, 0.5], [0.5, 0]],
                [[0, 0.5], [0, 0.5], [1, 0]],
            ]
        )

        # Get state tuple
        self.tuple_list_s = self.get_state_tuple()
        # Get action tuple
        self.tuple_list_a = self.get_action_tuple()

        # Create group list
        self.idx_list_d = range(self.n_group)
        # Create state list
        self.idx_list_s = range(len(self.tuple_list_s))
        # Create action list
        self.idx_list_a = range(len(self.tuple_list_a))

        # Create transition list
        self.bigT = self.generate_big_transition_matrix()
        # Create cost list
        self.bigC = self.generate_big_cost_matrix()

    def get_state_tuple(self):
        """A helper function used to get state tuple: cartesian s^D.

        Example (3 states, 2 groups):
            [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]

        """
        # Used to generate state cartesian product
        state_D_dim_temp = [list(range(self.n_state))] * self.n_group
        tuple_list_s = list(itertools.product(*state_D_dim_temp))
        return tuple_list_s

    def get_action_tuple(self):
        """ A helper function used to get action tuple list: [Keep] + [replace_1, ..., replace_D].

        Example (2 groups, 0: not replace, 1: replace):
            [0, 0; 1, 0; 0, 1]

        """
        # Keep all groups ([0]*D)
        tuple_list_a = [np.zeros(self.n_group, dtype=int).tolist()]
        # Replace the n-th group (diagonal[replace_1, ..., replace_D])
        tuple_list_a.extend(np.diag(np.ones(self.n_group, dtype=int)).tolist())
        return tuple_list_a

    def generate_big_cost_matrix(self):
        """Generate the cost matrix R(s, a, d) at state s taking action a for group d."""
        bigC = np.zeros(
            [len(self.tuple_list_s), len(self.tuple_list_a), len(self.idx_list_d)]
        )

        # Generates random immediate costs
        cost = np.zeros((self.n_state, self.n_action, self.n_group))
        for d in self.idx_list_d:
            # Keeps the machine
            operation_cost = [
                op_cost for op_cost in self.operation_cost
            ]  # + random.uniform(0, 5) * d
            cost[:, 0, d] = operation_cost
            # Replaces the machine (replace cost + new machine operation cost, no delivery lead time)
            cost[:, 1, d] = self.replace_cost + operation_cost[0]

        for s in self.idx_list_s:
            for a in self.idx_list_a:
                for d in self.idx_list_d:
                    bigC[s, a, d] = cost[
                        self.tuple_list_s[s][d], self.tuple_list_a[a][d], d
                    ]
        return bigC

    def generate_big_transition_matrix(self):
        """Generate the transition matrix Pr(s, s', a) from state s to state s' taking action a."""
        # matrix_T = self.transition_matrix
        bigT = np.zeros(
            [len(self.tuple_list_s), len(self.tuple_list_s), len(self.tuple_list_a)]
        )

        # Generates random transition matrix
        matrix_T = np.zeros((self.n_state, self.n_state, self.n_action, self.n_group))
        for d in self.idx_list_d:
            for s in range(self.n_state):
                temp_p = 0.5  # random.uniform(0.5, 0.8)
                if s < self.n_state - 1:
                    # Not replace
                    matrix_T[s, s, 0, d] = temp_p
                    matrix_T[s, s + 1, 0, d] = 1 - temp_p
                else:
                    # Not replace
                    matrix_T[s, s, 0, d] = 1
                # Replace
                matrix_T[s, 0, 1, d] = matrix_T[0, 0, 0, d]  # Stay in state 0
                matrix_T[s, 1, 1, d] = matrix_T[0, 1, 0, d]  # Transit to state 1

        for s in self.idx_list_s:
            for a in self.idx_list_a:
                for next_s in self.idx_list_s:
                    tmpT = 1
                    for d in self.idx_list_d:
                        tmpT *= matrix_T[
                            self.tuple_list_s[s][d],
                            self.tuple_list_s[next_s][d],
                            self.tuple_list_a[a][d],
                            d,
                        ]
                    bigT[s, next_s, a] = tmpT
        return bigT
