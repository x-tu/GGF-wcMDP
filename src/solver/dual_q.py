"""This script includes all the functions used to solve the dual Q model."""
from typing import Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def build_dual_q_model(q_values: list, weights: list) -> pyo.ConcreteModel:
    """The main function to build the dual Q model.

    Args:
        q_values (list): q_values[group][action], the q values for each group and action.
        weights (list): weights[group], the weights for each group.

    Returns:
        model (ConcreteModel): the pyomo model to solve.
    """

    model = pyo.ConcreteModel()

    # group index list
    idx_list_d = list(range(len(q_values)))
    # action index list
    idx_list_a = list(range(len(q_values[0])))

    # Variables
    # decision variable lambda
    model.varL = pyo.Var(idx_list_d, within=pyo.Reals)
    # decision variable nu
    model.varN = pyo.Var(idx_list_d, within=pyo.Reals)
    # decision variable pi(a|s)
    model.varP = pyo.Var(
        idx_list_a, within=pyo.NonNegativeReals, initialize=1 / len(idx_list_a)
    )

    # Objective
    model.cost = pyo.Objective(
        expr=sum(model.varL[d] for d in idx_list_d)
        + sum(model.varN[d] for d in idx_list_d),
        sense=pyo.maximize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (D * D Constraints)
    for d1 in idx_list_d:
        for d2 in idx_list_d:
            model.dual_constraints.add(
                model.varL[d1] + model.varN[d2]
                <= weights[d1]
                * sum(q_values[d2][a] * model.varP[a] for a in idx_list_a)
            )

    # Group 2 (1 constraint): the probabilities sum to 1
    model.dual_constraints.add(sum(model.varP[a] for a in idx_list_a) == 1)
    return model


def get_policy_from_q_values(q_values: list, weights: list) -> list:
    """Solve the model and get the policy based on the q values.

    Args:
        q_values (list): q_values[group][action], the q values for each group and action.
        weights (list): weights[group], the weights for each group.

    Returns:
        policy (list): policy[action], the optimal policy distribution for each action.
    """

    # build the model
    model = build_dual_q_model(q_values=q_values, weights=weights)
    # solve the model
    optimizer = SolverFactory("gurobi", solver_io="python")
    optimizer.solve(model, tee=False)
    # extract and return the policy distribution
    policy = [model.varP[a].value for a in model.varP]
    return policy


def test_deterministic_optimal(q_values: np.array, weights: list) -> Tuple[bool, int]:
    """Check whether the optimal policy is deterministic.

    Args:
        q_values (np.array): q_values[group][action], the q values for each group and action.
        weights (list): weights[group], the weights for each group.

    Returns:
        is_deterministic_optimal (bool): whether the optimal policy is deterministic.
        a_idx (int): the index of the action with the highest GGF Q value.
    """

    # calculate the GGF Q values
    ggf_q_values = np.dot(weights, np.sort(q_values, axis=0))
    # get the action with the highest GGF Q value
    a_idx = np.argmax(ggf_q_values).item()
    is_deterministic_optimal = ggf_q_values[a_idx] >= np.dot(
        weights, q_values[:, a_idx]
    )
    return is_deterministic_optimal, a_idx
