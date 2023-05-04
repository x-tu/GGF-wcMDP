from typing import List

import numpy as np


class MRPData:
    def __init__(self, n_group=2, n_state=3, n_action=2):
        self.n_group = n_group
        self.n_state = n_state
        self.n_action = n_action
        # TODO: implement more general cases
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

        # Get state list
        self.state_list = self.get_state_list()
        # Get action list
        self.action_list = self.get_action_list()

        # Create group index list
        self.idx_list_d = range(self.n_group)
        # Create state index list
        self.idx_list_s = range(len(self.state_list))
        # Create action index list
        self.idx_list_a = range(len(self.action_list))

        # Create transition matrix (s, s', a)
        self.bigT = self.generate_big_transition_matrix()
        # Create cost matrix (s, a, D)
        self.bigC = self.generate_big_cost_matrix()

    def get_state_list(self) -> List[List]:
        """ A helper function used to get state list: cartesian s^D.

        Example (3 states, 2 groups):
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

        Returns:
            state_list: state tuple list

        """
        # generate state indices
        state_indices = np.arange(self.n_state)
        # get cartesian product
        state_indices_cartesian = np.meshgrid(
            *([state_indices] * self.n_group), indexing="ij"
        )
        # reshape and convert to list
        state_list = (
            np.stack(state_indices_cartesian, axis=-1)
            .reshape(-1, self.n_group)
            .tolist()
        )
        return state_list

    def get_action_list(self) -> List[List]:
        """ A helper function used to get action list: [Keep] + [replace_1, ..., replace_D].

        Example (2 groups, 0: not replace, 1: replace):
            [[0, 0], [1, 0], [0, 1]]

        Returns:
            action_list: An action list of size (D+1) * D

        """
        # do nothing for all machines
        keep_action = np.zeros(self.n_group, dtype=int)
        # replace one machine
        replace_actions = np.eye(self.n_group, dtype=int)
        # combine and return
        action_list = np.vstack((keep_action, replace_actions)).tolist()
        return action_list

    def generate_big_cost_matrix(self) -> np.ndarray:
        """Generate the cost matrix R(s, a, d) at state s taking action a for group d.

        Returns:
            bigC: A cost matrix of size (S, A, D)

        """
        bigC = np.zeros(
            [len(self.state_list), len(self.action_list), len(self.idx_list_d)]
        )

        # Generates random immediate costs
        cost = np.zeros((self.n_state, self.n_action, self.n_group))
        for d in self.idx_list_d:
            # Keeps the machine
            operation_cost = [op_cost for op_cost in self.operation_cost]
            cost[:, 0, d] = operation_cost
            # Replaces the machine (replace cost + new machine operation cost, no delivery lead time)
            cost[:, 1, d] = self.replace_cost + operation_cost[0]

        for s in self.idx_list_s:
            for a in self.idx_list_a:
                for d in self.idx_list_d:
                    bigC[s, a, d] = cost[
                        self.state_list[s][d], self.action_list[a][d], d
                    ]
        return bigC

    def generate_big_transition_matrix(self):
        """Generate the transition matrix Pr(s, s', a) from state s to state s' taking action a.

        Returns:
            bigT: A transition matrix of size (S, S, A)

        """
        bigT = np.zeros(
            [len(self.state_list), len(self.state_list), len(self.action_list)]
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
                            self.state_list[s][d],
                            self.state_list[next_s][d],
                            self.action_list[a][d],
                            d,
                        ]
                    bigT[s, next_s, a] = tmpT
        return bigT
