import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import itertools
import numpy as np
from params_mrp import CostReward, MarkovChain, FairWeight


class LPData:
    def __init__(self, num_arms, num_states, rccc_wrt_max, prob_remain, mat_type, weight_coefficient, discount):
        self.num_arms = num_arms
        self.num_states = num_states
        self.num_actions = num_arms + 1
        self.rccc_wrt_max = rccc_wrt_max
        self.prob_remain = prob_remain
        self.mat_type = mat_type
        self.weight_coefficient = weight_coefficient
        self.state_tuples = self.get_state_tuples()
        self.action_tuples = self.get_action_tuples()
        # print(self.state_tuples)
        # print(self.action_tuples)
        # print('================')
        self.arm_indices = range(num_arms)
        self.state_indices = range(len(self.state_tuples))
        self.action_indices = range(len(self.action_tuples))
        rew_class = CostReward(self.num_states, self.num_arms, self.rccc_wrt_max)
        costs = rew_class.costs
        self.global_costs = self.get_global_costs(costs)
        dyn_class = MarkovChain(self.num_states, self.num_arms, self.prob_remain, self.mat_type)
        transitions = dyn_class.transitions
        self.global_transitions = self.get_global_transitions(transitions)
        wgh_class = FairWeight(num_arms, weight_coefficient)
        self.weights = wgh_class.weights
        self.discount = discount

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

    def get_global_costs(self, costs):
        """Generate the cost matrix R(s, a, d) at state s taking action a for group d."""
        global_costs = np.zeros([len(self.state_tuples), len(self.action_tuples), self.num_arms])
        for s in self.state_indices:
            for a in self.action_indices:
                for d in self.arm_indices:
                    global_costs[s, a, d] = costs[self.state_tuples[s][d], d, self.action_tuples[a][d]]
        return global_costs

    def get_global_transitions(self, transitions):
        """Generate the transition matrix Pr(s, s', a) from state s to state s' taking action a."""
        global_transitions = np.zeros([len(self.state_tuples), len(self.state_tuples), len(self.action_tuples)])

        for s in self.state_indices:
            for a in self.action_indices:
                for next_s in self.state_indices:
                    temp_trans = 1
                    for d in self.arm_indices:
                        temp_trans *= transitions[self.state_tuples[s][d],
                                                  self.state_tuples[next_s][d],
                                                  d,
                                                  self.action_tuples[a][d]]
                    global_transitions[s, next_s, a] = temp_trans
        return global_transitions


def build_lp(lp_data) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel()

    # Create mu list
    big_mu_list = [1 / len(lp_data.state_tuples)] * len(lp_data.state_tuples)

    # Variables
    model.varL = pyo.Var(lp_data.arm_indices, within=pyo.NonNegativeReals)
    model.varN = pyo.Var(lp_data.arm_indices, within=pyo.NonNegativeReals)
    model.varD = pyo.Var(
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
                    lp_data.global_costs[s, a, d2] * model.varD[lp_data.state_tuples[s], a]
                    for s in lp_data.state_indices
                    for a in lp_data.action_indices
                )
            )

    # Group 2 (s ^ D Constraints)
    for s in lp_data.state_indices:
        model.dual_constraints.add(
            sum(model.varD[lp_data.state_tuples[s], a] for a in lp_data.action_indices)
            - lp_data.discount
            * (
                sum(
                    model.varD[lp_data.state_tuples[next_s], a] * lp_data.global_transitions[s, next_s, a]
                    for next_s in lp_data.state_indices
                    for a in lp_data.action_indices
                )
            )
            == big_mu_list[s]
        )
    return model


def solve_lp(model):
    """ Selects the solver and set the optimization settings.

    Args:
        model: the MRP model to be optimized

    Returns:
        results: the default optimization report
        model: the optimized model

    """

    # Set the solver to be used
    optimizer = SolverFactory("gurobi", solver_io="python")
    # optimizer.options["sec"] = MAX_SOLVING_TIME
    results = optimizer.solve(model, tee=True)

    return results, model


def extract_lp(model: pyo.ConcreteModel, lp_data):
    """ This function is used to extract optimized results.

    Args:
        :param model: the optimized concrete model
        :param lp_data: the MRP parameter setting

    Returns:
        reward: the rewards for all groups
        policy: the policy to use

    """
    # Dual variable x
    for s in lp_data.state_indices:
        for a in lp_data.action_indices:
            x_value = model.varD[lp_data.state_tuples[s], a].value
            if x_value > 1e-6:
                print(f"x{lp_data.state_tuples[s], a}: {x_value}")

    # Policy interpretation
    # policy = np.zeros((9, 3))
    for s in lp_data.state_indices:
        x_sum = sum(
            [model.varD[lp_data.state_tuples[s], a].value for a in lp_data.action_indices]
        )
        for a in lp_data.action_indices:
            x_value = model.varD[lp_data.state_tuples[s], a].value
            if x_value > 1e-6:
                print(f"policy{lp_data.state_tuples[s], a}: {x_value / x_sum}")
                # policy[s, a] += x_value / x_sum

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
            lp_data.global_costs[s, a, d] * model.varD[lp_data.state_tuples[s], a].value
            for s in lp_data.state_indices
            for a in lp_data.action_indices
        )
        reward.append(all_cost)
        print(f"group {d}: {all_cost}")

    return reward  # , policy


def policy_lp(state, model: pyo.ConcreteModel, lp_data):
    for a in lp_data.action_indices:
        x_value = model.varD[tuple(state), a].value
        if x_value > 1e-6:
            action = int(a)

    return action
