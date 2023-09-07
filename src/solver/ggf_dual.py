"""This script includes all the functions used for solving GGF-MDP dual model."""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from utils.mrp import MRPData


def build_ggf(data: MRPData, solve_deterministic: bool = False) -> pyo.ConcreteModel:
    """The main function used to build the dual MRP concrete model.

    Args:
        data (`dict`): parameters used to solve the model
        solve_deterministic (`bool`): whether to solve the model deterministically

    Returns:
        model (`ConcreteModel`): the pyomo model to solve

    """
    model = pyo.ConcreteModel()

    # Create mu list
    big_mu_list = [1 / len(data.tuple_list_s)] * len(data.tuple_list_s)

    # Variables
    # decision variable lambda
    model.varL = pyo.Var(data.idx_list_d, within=pyo.NonNegativeReals)
    # decision variable nu
    model.varN = pyo.Var(data.idx_list_d, within=pyo.NonNegativeReals)
    # decision variable x
    model.varD = pyo.Var(
        data.tuple_list_s, data.idx_list_a, within=pyo.NonNegativeReals
    )
    if solve_deterministic:
        model.varP = pyo.Var(data.tuple_list_s, data.idx_list_a, within=pyo.Binary)

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
    # (skip for now) TODO: Group 3 (s ^D * D Constraints) whether to solve deterministically

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
    # policy = np.zeros((9, 3))
    for s in data.idx_list_s:
        x_sum = sum(
            [model.varD[data.tuple_list_s[s], a].value for a in data.idx_list_a]
        )
        for a in data.idx_list_a:
            x_value = model.varD[data.tuple_list_s[s], a].value
            if x_value > 1e-6:
                print(f"policy{data.tuple_list_s[s], a}: {x_value / x_sum}")
                # policy[s, a] += x_value / x_sum

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

    return reward  # , policy


def solve_ggf(input_data: MRPData, solve_deterministic: bool = False):
    """ Selects the solver and set the optimization settings.

    Args:
        input_data: the MRP parameter setting
        solve_deterministic: whether to solve deterministically

    Returns:
        results: the default optimization report
        model: the optimized model

    """
    # Build the GGF model
    model = build_ggf(data=input_data, solve_deterministic=solve_deterministic)
    # Set the solver to be used
    optimizer = SolverFactory("gurobi", solver_io="python")
    # optimizer.options["sec"] = MAX_SOLVING_TIME
    results = optimizer.solve(model, tee=True)
    # Extract the results
    reward = extract_results(model=model, data=input_data)
    # Extract the results
    extract_results(model=model, data=input_data)
    return results, model


def get_policy(state, model, data):
    """ This function is used to get the policy from the model given a state.

    Args:
        state: the current state
        model: the optimized model
        data: the MRP parameter setting

    Returns:
        a: the action to take

    """
    for a in data.idx_list_a:
        x_value = model.varD[data.tuple_list_s[state], a].value
        if x_value > 1e-6:
            return a
