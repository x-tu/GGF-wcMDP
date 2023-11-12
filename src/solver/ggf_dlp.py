import random
from datetime import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from utils.common import MDP4LP


def build_dlp(mdp: MDP4LP) -> pyo.ConcreteModel:
    """Used to build the GGF dual MDP (stochastic) model."""

    model = pyo.ConcreteModel()

    # Create mu list
    big_mu_list = [1 / len(mdp.state_indices)] * len(mdp.state_indices)

    # Variables
    model.varL = pyo.Var(mdp.group_indices.tolist(), within=pyo.Reals)
    model.varN = pyo.Var(mdp.group_indices, within=pyo.Reals)
    model.varX = pyo.Var(
        mdp.state_indices, mdp.action_indices, within=pyo.NonNegativeReals
    )

    # Objective
    model.objective = pyo.Objective(
        expr=sum(model.varL[d] for d in mdp.group_indices)
        + sum(model.varN[d] for d in mdp.group_indices),
        sense=pyo.minimize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 1 (D * D Constraints)
    for d1 in mdp.group_indices:
        for d2 in mdp.group_indices:
            model.dual_constraints.add(
                model.varL[d1] + model.varN[d2]
                >= mdp.weights[d1]
                * sum(
                    mdp.costs[s, a, d2] * model.varX[s, a]
                    for s in mdp.state_indices
                    for a in mdp.action_indices
                )
            )

    # Group 2 (s ^ D Constraints)
    for s in mdp.state_indices:
        model.dual_constraints.add(
            sum(model.varX[s, a1] for a1 in mdp.action_indices)
            - mdp.discount
            * (
                sum(
                    model.varX[j, a2] * mdp.transition[j, s, a2]
                    for j in mdp.state_indices
                    for a2 in mdp.action_indices
                )
            )
            == big_mu_list[s]
        )
    return model


def build_dlp_fix(mdp: MDP4LP, policy: pd.DataFrame) -> pyo.ConcreteModel:
    """Used to build the GGF dual MDP (stochastic) model with fixed policy.

    Args:
        mdp: the MRP parameter setting
        policy: the given policy to use
    """

    model = build_dlp(mdp=mdp)
    # Group 3 (s ^ D * D Constraints)
    for s in mdp.state_indices:
        for a in mdp.action_indices:
            model.dual_constraints.add(
                model.varX[s, a]
                == policy.iloc[s, a] * sum(model.varX[s, a] for a in mdp.action_indices)
            )
    return model


def solve_dlp(model: pyo.ConcreteModel):
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


def extract_dlp(model: pyo.ConcreteModel, mdp: MDP4LP):
    """ This function is used to extract optimized results.

    Args:
        :param model: the optimized concrete model
        :param mdp: the MRP parameter setting

    Returns:
        reward: the rewards for all groups
        policy: the policy to use

    """
    # Dual variable x
    # for s in mdp.state_indices:
    #     for a in mdp.action_indices:
    #         x_value = model.varX[mdp.state_tuples[s], a].value
    #         if x_value > 1e-6:
    #             print(f"x{mdp.state_tuples[s], a}: {x_value}")

    # Policy interpretation
    policy = {}
    x_total = 0
    for s in mdp.state_indices:
        x_sum = sum([model.varX[s, a].value for a in mdp.action_indices])
        x_total += x_sum
        x_sum = max(x_sum, 1e-6)  # avoid zero division

        action_prob = [model.varX[s, a].value / x_sum for a in mdp.action_indices]
        policy[s] = action_prob

        for a in mdp.action_indices:
            x_value = model.varX[s, a].value
            # if x_value > 1e-6:
            x_sum = 1e-6 if x_sum == 0 else x_sum  # avoid zero division
            print(f"policy{s, a}: {x_value / x_sum}")

    # Dual variable lambda
    for d in mdp.group_indices:
        print(f"lambda{d}: {model.varL[d].value}")

    # Dual variable nu
    for d in mdp.group_indices:
        print(f"nu{d}: {model.varN[d].value}")

    # Costs for group
    reward = []
    for d in mdp.group_indices:
        all_cost = sum(
            mdp.costs[s, a, d] * model.varX[s, a].value
            for s in mdp.state_indices
            for a in mdp.action_indices
        )
        reward.append(all_cost)
        print(f"group {d}: {all_cost}")
    reward = np.sort(np.array(reward))
    ggf_value = np.dot(reward, mdp.weights)
    print("x_total: ", x_total)
    print("GGF Value (DLP): ", ggf_value)
    return reward, policy, ggf_value


def policy_dlp(mdp, state, model: pyo.ConcreteModel, deterministic=False):
    if deterministic:
        for a in mdp.action_indices:
            x_value = model.varX[state, a].value
            if x_value > 1e-6:
                return int(a)
    else:
        x_values = [model.varX[state, int(a)].value for a in list(mdp.action_indices)]
        x_probs = [x / sum(x_values) for x in x_values]
        return random.choices(mdp.action_indices, weights=x_probs, k=1)[0]
