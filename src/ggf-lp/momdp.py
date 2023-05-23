"""This is an implementation for MOMDP."""

import pyomo.environ as pyo
from mrp_data import MRPData


def solve_mrp(data: MRPData) -> pyo.ConcreteModel:
    """The main function used to solve the MRP.

    Args:
        data (`dict`): parameters used to solve the model

    Returns:
        model (`ConcreteModel`): the pyomo model to solve

    """
    model = pyo.ConcreteModel()

    # Create mu list
    big_mu_list = [1 / len(data.tuple_list_s)] * len(data.tuple_list_s)

    # Variables
    model.varD = pyo.Var(
        data.tuple_list_s, data.idx_list_a, within=pyo.NonNegativeReals
    )

    # Objective
    model.cost = pyo.Objective(
        expr=sum(
            data.weight[d] * data.bigC[s, a, d] * model.varD[data.tuple_list_s[s], a]
            for s in data.idx_list_s
            for a in data.idx_list_a
            for d in data.idx_list_d
        ),
        sense=pyo.minimize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (s ^ D Constraints)
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


def extract_results(model: pyo.ConcreteModel, data: MRPData) -> list:
    """ This function is used to extract optimized results.

    Args:
        model: the optimized concrete model
        data: the MRP parameter setting

    Returns:
        reward: the rewards for all groups

    """
    # Dual variable x
    for s in data.idx_list_s:
        for a in data.idx_list_a:
            x_value = model.varD[data.tuple_list_s[s], a].value
            if x_value > 1e-6:
                print(f"x{data.tuple_list_s[s], a}: {x_value}")

    # Policy interpretation
    for s in data.idx_list_s:
        x_sum = sum(
            [model.varD[data.tuple_list_s[s], a].value for a in data.idx_list_a]
        )
        for a in data.idx_list_a:
            x_value = model.varD[data.tuple_list_s[s], a].value
            if x_value > 1e-6:
                print(f"policy{data.tuple_list_s[s], a}: {x_value / x_sum}")

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

    return reward


# Get data
input_data = MRPData(n_group=3, n_state=3, n_action=2, weight=[1, 0.5, 0.25])
ggf_model = solve_mrp(data=input_data)
pyo.SolverFactory("cbc").solve(ggf_model).write()
extract_results(model=ggf_model, data=input_data)
