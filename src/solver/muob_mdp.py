import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def build_mlp(data) -> pyo.ConcreteModel:
    """The main function used to build the MRP model.

    Args:
        data (`dict`): parameters used to solve the model

    Returns:
        model (`ConcreteModel`): the pyomo model to solve

    """
    model = pyo.ConcreteModel()

    # Create mu list
    big_mu_list = [1 / len(data.state_tuples)] * len(data.state_tuples)

    # Variables
    model.varD = pyo.Var(
        data.state_tuples, data.action_indices, within=pyo.NonNegativeReals
    )

    # Objective
    model.cost = pyo.Objective(
        expr=sum(
            data.weights[d]
            * data.global_costs[s, a, d]
            * model.varD[data.state_tuples[s], a]
            for s in data.state_indices
            for a in data.action_indices
            for d in data.arm_indices
        ),
        sense=pyo.minimize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (s ^ D Constraints)
    for s in data.state_indices:
        model.dual_constraints.add(
            sum(model.varD[data.state_tuples[s], a] for a in data.action_indices)
            - data.discount
            * (
                sum(
                    model.varD[data.state_tuples[next_s], a]
                    * data.global_transitions[s, next_s, a]
                    for next_s in data.state_indices
                    for a in data.action_indices
                )
            )
            == big_mu_list[s]
        )
    return model


def extract_mlp(model: pyo.ConcreteModel, data) -> list:
    """ This function is used to extract optimized results.

    Args:
        model: the optimized concrete model
        data: the MRP parameter setting

    Returns:
        reward: the rewards for all groups

    """
    # Dual variable x
    for s in data.state_indices:
        for a in data.action_indices:
            x_value = model.varD[data.state_tuples[s], a].value
            # if x_value > 1e-6:
            #     print(f"x{data.state_tuples[s], a}: {x_value}")

    # Policy interpretation
    for s in data.state_indices:
        x_sum = sum(
            [model.varD[data.state_tuples[s], a].value for a in data.action_indices]
        )
        for a in data.action_indices:
            x_value = model.varD[data.state_tuples[s], a].value
            # if x_value > 1e-6:
            #     print(f"policy{data.state_tuples[s], a}: {x_value / x_sum}")

    # Costs for group
    reward = []
    for d in data.arm_indices:
        all_cost = sum(
            data.global_costs[s, a, d] * model.varD[data.state_tuples[s], a].value
            for s in data.state_indices
            for a in data.action_indices
        )
        reward.append(all_cost)
        print(f"group {d}: {all_cost}")

    return reward


def solve_mlp(model):
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


def policy_mlp(state, model: pyo.ConcreteModel, lp_data):
    for a in lp_data.action_indices:
        x_value = model.varD[tuple(state), a].value
        if x_value > 1e-6:
            action = int(a)

    return action
