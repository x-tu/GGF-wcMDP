"""This is an implementation for GGF-MDP dual form."""

import numpy as np
import pyomo.environ as pyo
from mrp_data import MRPData
from pyomo.opt import SolverFactory


def build_ggf(data: MRPData) -> pyo.ConcreteModel:
    """The main function used to build the dual MRP concrete model.

    Args:
        data (`dict`): parameters used to solve the model

    Returns:
        model (`ConcreteModel`): the pyomo model to solve

    """
    model = pyo.ConcreteModel()

    # Create mu list
    big_mu_list = [1 / len(data.tuple_list_s)] * len(data.tuple_list_s)

    # Variables
    model.varL = pyo.Var(data.idx_list_d, within=pyo.NonNegativeReals)
    model.varN = pyo.Var(data.idx_list_d, within=pyo.NonNegativeReals)
    model.varD = pyo.Var(
        data.tuple_list_s, data.idx_list_a, within=pyo.NonNegativeReals
    )

    # Objective
    model.cost = pyo.Objective(
        expr=sum(model.varL[d] for d in data.idx_list_d)
        + sum(model.varN[d] for d in data.idx_list_d),
        sense=pyo.minimize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (D * D Constraints)
    for d1 in data.idx_list_d:
        for d2 in data.idx_list_d:
            model.dual_constraints.add(
                model.varL[d1] + model.varN[d2]
                >= data.weight[d1]
                * sum(
                    data.bigC[s, a, d2] * model.varD[data.tuple_list_s[s], a]
                    for s in data.idx_list_s
                    for a in data.idx_list_a
                )
            )

    # Group 2 (s ^ D Constraints)
    for s in data.idx_list_s:
        model.dual_constraints.add(
            sum(model.varD[data.tuple_list_s[s], a] for a in data.idx_list_a)
            - data.discount
            * (
                sum(
                    model.varD[data.tuple_list_s[next_s], a] * data.bigT[s, next_s, a]
                    for next_s in data.idx_list_s
                    for a in data.idx_list_a
                )
            )
            == big_mu_list[s]
        )
    return model


def extract_results(model: pyo.ConcreteModel, data: MRPData):
    """ This function is used to extract optimized results.

    Args:
        model: the optimized concrete model
        data: the MRP parameter setting

    Returns:
        reward: the rewards for all groups
        policy: the policy to use

    """
    # Dual variable x
    for s in data.idx_list_s:
        for a in data.idx_list_a:
            x_value = model.varD[data.tuple_list_s[s], a].value
            if x_value > 1e-6:
                print(f"x{data.tuple_list_s[s], a}: {x_value}")

    # Policy interpretation
    policy = np.zeros((9, 3))
    for s in data.idx_list_s:
        x_sum = sum(
            [model.varD[data.tuple_list_s[s], a].value for a in data.idx_list_a]
        )
        for a in data.idx_list_a:
            x_value = model.varD[data.tuple_list_s[s], a].value
            if x_value > 1e-6:
                print(f"policy{data.tuple_list_s[s], a}: {x_value / x_sum}")
                policy[s, a] += x_value / x_sum

    # Dual variable lambda
    for d in data.idx_list_d:
        print(f"lambda{d}: {model.varL[d].value}")

    # Dual variable nu
    for d in data.idx_list_d:
        print(f"nu{d}: {model.varN[d].value}")

    # Costs for group
    reward = []
    for d in data.idx_list_d:
        all_cost = sum(
            data.bigC[s, a, d] * model.varD[data.tuple_list_s[s], a].value
            for s in data.idx_list_s
            for a in data.idx_list_a
        )
        reward.append(all_cost)
        print(f"group {d}: {all_cost}")

    return reward, policy


def solve_ggf(model):
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


# Get data
input_data = MRPData(n_group=2, n_state=3, n_action=2, weight=[0.5, 0.5])

ggf_model = build_ggf(data=input_data)
results, ggf_model = solve_ggf(model=ggf_model)
extract_results(model=ggf_model, data=input_data)
