"""This is an implementation for GGF-LP."""

import itertools

import numpy as np
import pyomo.environ as pyo

# Parameters
input_data = {
    "D_GROUP": 2,
    # TODO: sort the weights
    "WEIGHT_D": [0.5, 0.5],
    "N_STATE": 3,
    "N_ACTION": 2,
    "OPERATION_COST": [10, 20, 50],
    "REPLACE_COST": 100,
    "DISCOUNT": 0.9,
    "TRANSITION_MATRIX": np.array([[[0.5, 0.5], [0.5, 0.5], [0, 0]],
                                   [[0, 0.5], [0.5, 0.5], [0.5, 0]],
                                   [[0, 0.5], [0, 0.5], [1, 0]]])
}


def solve_ggf(data: dict) -> pyo.ConcreteModel:
    """The main function used to solve the dual MRP.

    Args:
        data (`dict`): parameters used to solve the model

    Returns:
        model (`ConcreteModel`): the pyomo model to solve

    """
    model = pyo.ConcreteModel()

    # Index
    # Create group list
    d_idx_list = range(data["D_GROUP"])
    # Create state list: cartesian s^D
    # Used to generate state cartesian product
    state_D_dim_temp = [list(range(data["N_STATE"]))] * data["D_GROUP"]
    s_tuple_list = list(itertools.product(*state_D_dim_temp))
    s_idx_list = range(len(s_tuple_list))
    # Create action list: [Keep] + [replace_1, ..., replace_D]
    a_tuple_list = [np.zeros(data["D_GROUP"], dtype=int).tolist()]
    a_tuple_list.extend(np.diag(np.ones(data["D_GROUP"], dtype=int)).tolist())
    a_idx_list = range(len(a_tuple_list))

    # Parameters
    # Helper function 1
    def generate_big_cost_matrix():
        """Generate the cost matrix R(s, a, d) at state s taking action a for group d."""
        bigC = np.zeros([len(s_tuple_list), len(a_tuple_list), len(d_idx_list)])

        # Calculate immediate cost for a single machine
        # TODO: Change the rewards between different groups
        immediate_cost = np.zeros((data["N_STATE"], data["N_ACTION"]))
        # Keeps the machine
        immediate_cost[:, 0] = data["OPERATION_COST"]
        # Replaces the machine (replace cost + new machine operation cost, no delivery lead time)
        immediate_cost[:, 1] = data["REPLACE_COST"] + data["OPERATION_COST"][0]

        for s in s_idx_list:
            for a in a_idx_list:
                for d in d_idx_list:
                    bigC[s, a, d] = immediate_cost[s_tuple_list[s][d], a_tuple_list[a][d]]
        return bigC

    # Helper function 2
    def generate_big_transition_matrix():
        """Generate the transition matrix Pr(s, s', a) from state s to state s' taking action a."""
        matrix_T = data["TRANSITION_MATRIX"]
        bigT = np.zeros([len(s_tuple_list), len(s_tuple_list), len(a_tuple_list)])

        for s in s_idx_list:
            for a in a_idx_list:
                for next_s in s_idx_list:
                    tmpT = 1
                    for d in d_idx_list:
                        tmpT *= matrix_T[s_tuple_list[s][d], s_tuple_list[next_s][d], a_tuple_list[a][d]]
                    bigT[s, next_s, a] = tmpT
        return bigT

    # Create mu list
    big_mu_list = [1 / len(s_tuple_list)] * len(s_tuple_list)
    # Create reward list
    global big_cost
    big_cost = generate_big_cost_matrix()
    # Create transition list
    big_transition = generate_big_transition_matrix()

    # Variables
    model.varL = pyo.Var(d_idx_list, within=pyo.NonNegativeReals)
    model.varN = pyo.Var(d_idx_list, within=pyo.NonNegativeReals)
    model.varD = pyo.Var(s_tuple_list, a_idx_list, within=pyo.NonNegativeReals)

    # Objective
    model.cost = pyo.Objective(
        expr=sum(model.varL[d] for d in d_idx_list) + sum(model.varN[d] for d in d_idx_list), sense=pyo.minimize)

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (D * D Constraints)
    for d1 in d_idx_list:
        for d2 in d_idx_list:
            model.dual_constraints.add(model.varL[d1] + model.varN[d2] >= data["WEIGHT_D"][d1] * sum(
                big_cost[s, a, d2] * model.varD[s_tuple_list[s], a]
                for s in s_idx_list for a in a_idx_list))

    # Group 2 (s ^ D Constraints)
    for s in s_idx_list:
        model.dual_constraints.add(
            sum(model.varD[s_tuple_list[s], a] for a in a_idx_list)
            - data["DISCOUNT"] * (sum(model.varD[s_tuple_list[next_s], a] * big_transition[s, next_s, a]
                                      for next_s in s_idx_list for a in a_idx_list)) == big_mu_list[s]
        )
    return model


def extract_results(model: pyo.ConcreteModel, data: dict) -> None:
    # Index
    # Create group list
    d_idx_list = range(data["D_GROUP"])
    # Create state list: cartesian s^D
    # Used to generate state cartesian product
    state_D_dim_temp = [list(range(data["N_STATE"]))] * data["D_GROUP"]
    s_tuple_list = list(itertools.product(*state_D_dim_temp))
    s_idx_list = range(len(s_tuple_list))
    # Create action list: [Keep] + [replace_1, ..., replace_D]
    a_tuple_list = [np.zeros(data["D_GROUP"], dtype=int).tolist()]
    a_tuple_list.extend(np.diag(np.ones(data["D_GROUP"], dtype=int)).tolist())
    a_idx_list = range(len(a_tuple_list))

    # Dual variable x
    for s in s_idx_list:
        for a in a_idx_list:
            x_value = model.varD[s_tuple_list[s], a].value
            if x_value > 1e-6:
                print(f"x{s_tuple_list[s], a}: {x_value}")

    # Dual variable lambda
    for d in d_idx_list:
        print(f"lambda{d}: {model.varL[d].value}")

    # Dual variable nu
    for d in d_idx_list:
        print(f"nu{d}: {model.varN[d].value}")

    # Costs for group
    for d in d_idx_list:
        all_cost = sum(big_cost[s, a, d] * model.varD[s_tuple_list[s], a].value
                       for s in s_idx_list for a in a_idx_list)
        print(f"group {d}: {all_cost}")


ggf_model = solve_ggf(data=input_data)
pyo.SolverFactory('cbc').solve(ggf_model).write()
extract_results(model=ggf_model, data=input_data)
print(ggf_model)
