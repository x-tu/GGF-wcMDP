import itertools
import random
from datetime import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from utils.params_mrp import CostReward, MarkovChain


class LPData:
    def __init__(
        self,
        num_arms,
        num_states,
        rccc_wrt_max,
        prob_remain,
        mat_type,
        weights,
        discount,
        encoding_int=False,
    ):
        self.num_arms = num_arms
        self.num_states = num_states
        self.num_actions = num_arms + 1
        self.rccc_wrt_max = rccc_wrt_max
        self.prob_remain = prob_remain
        self.mat_type = mat_type
        self.weights = weights
        self.discount = discount
        self.state_tuples = self.get_state_tuples()
        self.action_tuples = self.get_action_tuples()
        # print(self.state_tuples)
        # print(self.action_tuples)
        # print('================')
        self.arm_indices = range(num_arms)
        self.state_indices = range(len(self.state_tuples))
        self.action_indices = range(len(self.action_tuples))
        rew_class = CostReward(self.num_states, self.num_arms, self.rccc_wrt_max)
        self.rewards = rew_class.rewards
        self.costs = rew_class.costs
        self.global_costs = self.get_global_costs()
        dyn_class = MarkovChain(
            self.num_states, self.num_arms, self.prob_remain, self.mat_type
        )
        self.transitions = dyn_class.transitions
        self.global_transitions = self.get_global_transitions()
        self.encoding_int = encoding_int
        if self.encoding_int:
            self.weight = self.weights
            # Get state tuple
            self.tuple_list_s = self.state_tuples
            # Get action tuple
            self.tuple_list_a = self.action_tuples

            # Create group list
            self.idx_list_d = self.arm_indices
            # Create state list
            self.idx_list_s = self.state_indices
            # Create action list
            self.idx_list_a = self.action_indices
            self.bigC = self.costs
            self.bigT = self.global_transitions

    def get_state_tuples(self):
        """A helper function used to get state tuple: cartesian S ** D.

        Example (3 states, 2 groups):
            [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]

        """
        state_dim_temp = [list(range(self.num_states))] * self.num_arms
        tuple_list_s = list(itertools.product(*state_dim_temp))
        return tuple_list_s

    def get_action_tuples(self):
        """ A helper function used to get action tuple list: [Keep] + [replace_1, ..., replace_D].

        Example (2 groups, 0: not replace, 1: replace):
            [0, 0; 1, 0; 0, 1]

        """
        # Keep all groups ([0]*D)
        tuple_list_a = [list(np.zeros(self.num_arms, dtype=int))]
        # Replace the n-th group (diagonal[replace_1, ..., replace_D])
        tuple_list_a.extend(np.diag(np.ones(self.num_arms, dtype=int)).tolist())
        return tuple_list_a

    def get_global_costs(self):
        """Generate the cost matrix R(s, a, d) at state s taking action a for group d."""
        global_costs = np.zeros(
            [len(self.state_tuples), len(self.action_tuples), self.num_arms]
        )
        for s in self.state_indices:
            for a in self.action_indices:
                for d in self.arm_indices:
                    global_costs[s, a, d] = self.costs[
                        self.state_tuples[s][d], d, self.action_tuples[a][d]
                    ]
        return global_costs

    def get_global_transitions(self):
        """Generate the transition matrix Pr(s, s', a) from state s to state s' taking action a."""
        global_transitions = np.zeros(
            [len(self.state_tuples), len(self.state_tuples), len(self.action_tuples)]
        )

        for s in self.state_indices:
            for a in self.action_indices:
                for next_s in self.state_indices:
                    temp_trans = 1
                    for d in self.arm_indices:
                        temp_trans *= self.transitions[
                            self.state_tuples[s][d],
                            self.state_tuples[next_s][d],
                            d,
                            self.action_tuples[a][d],
                        ]
                    global_transitions[s, next_s, a] = temp_trans
        return global_transitions


def build_dlp(lp_data: LPData, init_state_idx: int = None) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel()

    # Create mu list
    if not init_state_idx:
        big_mu_list = [1 / len(lp_data.state_tuples)] * len(lp_data.state_tuples)
    else:
        big_mu_list = [0.001] * len(lp_data.state_tuples)
        big_mu_list[init_state_idx] = 1 - 0.001 * (len(lp_data.state_tuples) - 1)

    # Variables
    model.varL = pyo.Var(lp_data.arm_indices, within=pyo.NonNegativeReals)
    model.varN = pyo.Var(lp_data.arm_indices, within=pyo.NonNegativeReals)
    model.varX = pyo.Var(
        lp_data.state_tuples, lp_data.action_indices, within=pyo.NonNegativeReals
    )

    # Objective
    model.cost = pyo.Objective(
        expr=sum(model.varL[d] for d in lp_data.arm_indices)
        + sum(model.varN[d] for d in lp_data.arm_indices),
        sense=pyo.minimize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (D * D Constraints)
    for d1 in lp_data.arm_indices:
        for d2 in lp_data.arm_indices:
            model.dual_constraints.add(
                model.varL[d1] + model.varN[d2]
                >= lp_data.weights[d1]
                * sum(
                    lp_data.global_costs[s, a, d2]
                    * model.varX[lp_data.state_tuples[s], a]
                    for s in lp_data.state_indices
                    for a in lp_data.action_indices
                )
            )

    # Group 2 (s ^ D Constraints)
    for s in lp_data.state_indices:
        model.dual_constraints.add(
            sum(
                model.varX[lp_data.state_tuples[s], a1] for a1 in lp_data.action_indices
            )
            - lp_data.discount
            * (
                sum(
                    model.varX[lp_data.state_tuples[j], a2]
                    * lp_data.global_transitions[j, s, a2]
                    for j in lp_data.state_indices
                    for a2 in lp_data.action_indices
                )
            )
            == big_mu_list[s]
        )
    return model


def build_dlp_fix(lp_data: LPData, policy: pd.DataFrame) -> pyo.ConcreteModel:
    """Build the fixed policy MRP model.

    Args:
        lp_data: the MRP parameter setting
        policy: the given policy to use
    """

    model = build_dlp(lp_data=lp_data)
    # Group 3 (s ^ D * D Constraints)
    for s in lp_data.state_indices:
        for a in lp_data.action_indices:
            model.dual_constraints.add(
                model.varX[lp_data.state_tuples[s], a]
                == policy.iloc[s, a]
                * sum(
                    model.varX[lp_data.state_tuples[s], a]
                    for a in lp_data.action_indices
                )
            )
    return model


def solve_dlp(model):
    """ Selects the solver and set the optimization settings.

    Args:
        model: the MRP model to be optimized

    Returns:
        results: the default optimization report
        model: the optimized model

    """

    # Set the solver to be used
    start_time = datetime.now()
    optimizer = SolverFactory("gurobi", solver_io="python")
    print(f"Solver selection time: {(datetime.now() - start_time).total_seconds()}")
    # optimizer.options["sec"] = MAX_SOLVING_TIME
    start_time = datetime.now()
    results = optimizer.solve(model, tee=False, report_timing=True)
    print(f"Solver solving time: {(datetime.now() - start_time).total_seconds()}")
    return results, model


def extract_dlp(model: pyo.ConcreteModel, lp_data):
    """ This function is used to extract optimized results.

    Args:
        :param model: the optimized concrete model
        :param lp_data: the MRP parameter setting

    Returns:
        reward: the rewards for all groups
        policy: the policy to use

    """
    # Dual variable x
    # for s in lp_data.state_indices:
    #     for a in lp_data.action_indices:
    #         x_value = model.varX[lp_data.state_tuples[s], a].value
    #         if x_value > 1e-6:
    #             print(f"x{lp_data.state_tuples[s], a}: {x_value}")

    # Policy interpretation
    policy = {}
    x_total = 0
    for s in lp_data.state_indices:
        x_sum = sum(
            [
                model.varX[lp_data.state_tuples[s], a].value
                for a in lp_data.action_indices
            ]
        )
        x_total += x_sum
        x_sum = max(x_sum, 1e-6)  # avoid zero division

        action_prob = [
            model.varX[lp_data.state_tuples[s], a].value / x_sum
            for a in lp_data.action_indices
        ]
        policy[s] = action_prob

        for a in lp_data.action_indices:
            x_value = model.varX[lp_data.state_tuples[s], a].value
            # if x_value > 1e-6:
            x_sum = 1e-6 if x_sum == 0 else x_sum  # avoid zero division
            print(f"policy{lp_data.state_tuples[s], a}: {x_value / x_sum}")

    # Dual variable lambda
    for d in lp_data.arm_indices:
        print(f"lambda{d}: {model.varL[d].value}")

    # Dual variable nu
    for d in lp_data.arm_indices:
        print(f"nu{d}: {model.varN[d].value}")

    # Costs for group
    reward = []
    for d in lp_data.arm_indices:
        all_cost = sum(
            lp_data.global_costs[s, a, d] * model.varX[lp_data.state_tuples[s], a].value
            for s in lp_data.state_indices
            for a in lp_data.action_indices
        )
        reward.append(all_cost)
        print(f"group {d}: {all_cost}")
    reward = np.sort(np.array(reward))
    ggf_value = np.dot(reward, lp_data.weights)
    print("x_total: ", x_total)
    print("GGF Value (DLP): ", ggf_value)
    return reward, policy, ggf_value


def policy_dlp(state, model: pyo.ConcreteModel, lp_data, deterministic=False):
    if deterministic:
        for a in lp_data.action_indices:
            x_value = model.varX[tuple(state), a].value
            if x_value > 1e-6:
                return int(a)
    else:
        x_values = [
            model.varX[tuple(state), int(a)].value for a in list(lp_data.action_indices)
        ]
        x_probs = [x / sum(x_values) for x in x_values]
        return random.choices(lp_data.action_indices, weights=x_probs, k=1)[0]
