"""This is used as a practice to get familiar with pyomo on solving MDP LP model.
    Examples are from DO weekly assignment 3 - MDP
"""

import numpy as np
import pyomo.environ as pyo

# Parameters
data = {
    "N_STATE": 3,
    "N_ACTION": 2,
    "OPERATION_COST": [10, 20, 50],
    "REPLACE_COST": 100,
    "DISCOUNT": 0.9,
    "TRANSITION_MATRIX": np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]]),
}


def solve_mrp(data: dict) -> pyo.ConcreteModel:
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

    # Variables
    model.varD = pyo.Var(state_list, action_list, within=pyo.NonNegativeReals)

    # Calculate immediate cost
    immediate_cost = np.zeros((data["N_STATE"], data["N_ACTION"]))
    # Keeps the machine
    immediate_cost[:, 0] = data["OPERATION_COST"]
    # Replaces the machine (replace cost + new machine operation cost, no delivery lead time)
    immediate_cost[:, 1] = data["REPLACE_COST"] + data["OPERATION_COST"][0]

    # Objective
    model.cost = pyo.Objective(
        expr=sum(immediate_cost[i, j] * model.varD[i, j] for i in state_list for j in action_list), sense=pyo.minimize)

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    for i in state_list:
        model.dual_constraints.add(
            sum(model.varD[i, j] for j in action_list) - data["DISCOUNT"] * (sum(
                model.varD[k, 0] * data["TRANSITION_MATRIX"][k, i] +
                model.varD[k, 1] * data["TRANSITION_MATRIX"][0, i] for k in state_list
            ))
            == 1
        )
    return model


model = solve_mrp(data)
pyo.SolverFactory('cbc').solve(model).write()
