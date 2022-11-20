"""This is an implementation for GGF-LP.
"""

import numpy as np
import pyomo.environ as pyo

# Parameters
input_data = {
    "D_GROUP": 2,
    "WEIGHT_D": [0.6, 0.4],
    "N_STATE": 3,
    "N_ACTION": 2,
    "OPERATION_COST": [10, 20, 50],
    "REPLACE_COST": 100,
    "DISCOUNT": 0.9,
    "TRANSITION_MATRIX": np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]]),
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
    state_list = range(data["N_STATE"])
    action_list = range(data["N_ACTION"])
    group_list = range(data["D_GROUP"])

    # Variables
    model.varL = pyo.Var(group_list, within=pyo.NonNegativeReals)
    model.varN = pyo.Var(group_list, within=pyo.NonNegativeReals)
    model.varD = pyo.Var(group_list, state_list, action_list, within=pyo.NonNegativeReals)
    model.varM = pyo.Var(state_list, within=pyo.NonNegativeReals)

    # Calculate immediate cost
    # TODO: Change the rewards between different groups
    immediate_cost = np.zeros((data["N_STATE"], data["N_ACTION"]))
    # Keeps the machine
    immediate_cost[:, 0] = data["OPERATION_COST"]
    # Replaces the machine (replace cost + new machine operation cost, no delivery lead time)
    immediate_cost[:, 1] = data["REPLACE_COST"] + data["OPERATION_COST"][0]

    # Objective
    model.cost = pyo.Objective(
        expr=sum(model.varL[d] for d in group_list) + sum(model.varN[d] for d in group_list), sense=pyo.minimize)

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (D * D Constraints)
    for d1 in group_list:
        for d2 in group_list:
            model.dual_constraints.add(model.varL[d1] + model.varN[d2] >= data["WEIGHT_D"][d1] * sum(
                immediate_cost[s, a] * model.varD[d1, s, a] for s in state_list for a in action_list))

    # Group 2 (s * D Constraints)
    for s in state_list:
        for d in group_list:
            model.dual_constraints.add(
                sum(model.varD[d, s, a] for a in action_list) - data["DISCOUNT"] * (sum(
                    # Keep the machine
                    model.varD[d, next_s, 0] * data["TRANSITION_MATRIX"][next_s, 0] +
                    # Replace the machine
                    model.varD[d, next_s, 1] * data["TRANSITION_MATRIX"][next_s, 1]
                    for next_s in state_list
                ))
                == model.varM[s]
            )

    # Group 3 (1 Constraint)
    model.dual_constraints.add(sum(model.varM[s] for s in state_list) == 1)
    return model


def extract_results(ggf_model: pyo.ConcreteModel, data: dict) -> None:
    # Index
    state_list = range(data["N_STATE"])
    action_list = range(data["N_ACTION"])
    group_list = range(data["D_GROUP"])

    # Print Results
    # Dual variable x
    for d in group_list:
        print(f"Group {d}")
        for s in state_list:
            for a in action_list:
                print(f"x{d, s, a}: {ggf_model.varD._data[d, s, a].value}")

    # Dual variable lambda
    for d in group_list:
        print(f"lambda{d}: {ggf_model.varL._data[d].value}")

    # Dual variable nu
    for d in group_list:
        print(f"nu{d}: {ggf_model.varN._data[d].value}")


ggf_model = solve_ggf(input_data)
pyo.SolverFactory('cbc').solve(ggf_model).write()
extract_results(ggf_model, input_data)
print(ggf_model)
