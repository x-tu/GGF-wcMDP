import random
from datetime import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from utils.common import MDP4LP, DotDict


def build_dlp(
    mdp: MDP4LP, deterministic_policy: bool = False, print_results: bool = False
) -> pyo.ConcreteModel:
    """Used to build the GGF dual MDP (stochastic) model."""

    model = pyo.ConcreteModel()
    model.deterministic_policy = deterministic_policy
    model.mdp = mdp
    model.print_results = print_results

    # Create mu list
    big_mu_list = [1 / len(mdp.state_indices)] * len(mdp.state_indices)

    # Variables
    model.varL = pyo.Var(mdp.group_indices.tolist(), within=pyo.Reals)
    model.varN = pyo.Var(mdp.group_indices, within=pyo.Reals)
    model.varX = pyo.Var(
        mdp.state_indices, mdp.action_indices, within=pyo.NonNegativeReals
    )
    if deterministic_policy:
        model.varPi = pyo.Var(mdp.state_indices, mdp.action_indices, domain=pyo.Binary)

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

    if deterministic_policy:
        # Group 4 - transition freq (s ^ D * D Constraints)
        M = 1 / (1 - mdp.discount)
        for s in mdp.state_indices:
            for a in mdp.action_indices:
                model.dual_constraints.add(model.varX[s, a] <= model.varPi[s, a] * M)

        # Group 5 - policy prob (s ^ D Constraints)
        for s in mdp.state_indices:
            model.dual_constraints.add(
                sum(model.varPi[s, a] for a in mdp.action_indices) == 1
            )

    return model


def build_dlp_fix(mdp: MDP4LP, policy: pd.DataFrame) -> pyo.ConcreteModel:
    """Used to build the GGF dual MDP (stochastic) model with fixed policy.

    Args:
        mdp: the MRP parameter setting
        policy: the given policy to use
    """

    model = build_dlp(mdp=mdp, deterministic_policy=False)
    # Group 0 (s ^ D * D Constraints)
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
    results = optimizer.solve(model, tee=False, report_timing=False)
    print(f"Solver solving time: {(datetime.now() - start_time).total_seconds()}")
    return results, model


def extract_dlp(model: pyo.ConcreteModel):
    """ This function is used to extract optimized results.

    Args:
        :param model: the optimized concrete model
        :param mdp: the MRP parameter setting

    Returns:
        reward: the rewards for all groups
        policy: the policy to use
    """

    # Policy formatting
    if model.deterministic_policy:
        # in case of the float zero error, format the policy to integer
        for s in model.mdp.state_indices:
            for a in model.mdp.action_indices:
                model.varPi[s, a] = round(model.varPi[s, a].value)
                # model.varX[s, a] = model.varPi[s, a] * model.varX[s, a]
            assert sum([model.varPi[s, a].value for a in model.mdp.action_indices]) == 1

    # Print results
    policy = {}
    var_x = {}
    x_total = 0
    for s in model.mdp.state_indices:
        state = model.mdp.state_tuple_list[s] if not model.mdp.encoding_int else s

        # used to calculate the total x
        x_sum = sum([model.varX[s, a].value for a in model.mdp.action_indices])
        x_sum = max(x_sum, 1e-6)  # avoid zero division

        # record the policy and visitation frequency
        policy[str(state)] = [
            model.varX[s, a].value / x_sum for a in model.mdp.action_indices
        ]
        var_x[str(state)] = [model.varX[s, a].value for a in model.mdp.action_indices]

        # print the policy if the visitation frequency is not zero
        for a in model.mdp.action_indices:
            x_value = model.varX[s, a].value
            if x_value > 1e-6 and model.print_results:
                print(f"policy{state, a}: {x_value / x_sum}")

        # calculate the total visitation frequency
        x_total += x_sum
    print("x_total: ", x_total) if model.print_results else None

    # Dual variable lambda
    var_lambda = {}
    for d in model.mdp.group_indices:
        var_lambda[str(d)] = model.varN[d].value
        print(f"lambda{d}: {model.varL[d].value}") if model.print_results else None

    # Dual variable nu
    var_nu = {}
    for d in model.mdp.group_indices:
        var_nu[str(d)] = model.varN[d].value
        print(f"nu{d}: {model.varN[d].value}") if model.print_results else None

    # Costs for group
    reward = []
    for d in model.mdp.group_indices:
        all_cost = sum(
            model.mdp.costs[s, a, d] * model.varX[s, a].value
            for s in model.mdp.state_indices
            for a in model.mdp.action_indices
        )
        reward.append(all_cost)
        print(f"group {d}: {all_cost}") if model.print_results else None

    # calculate the GGF value (XR)
    reward_sorted = np.sort(np.array(reward))
    ggf_value_xr = np.dot(reward_sorted, model.mdp.weights)
    print("GGF Value (DLP) XR: ", ggf_value_xr)

    # calculate the GGF value (L+N)
    ggf_value_ln = sum(
        model.varL[d].value + model.varN[d].value for d in model.mdp.group_indices
    )
    print("GGF Value (DLP) L+N: ", ggf_value_ln)

    results = DotDict(
        {
            "policy": policy,
            "var_x": var_x,
            "var_lambda": var_lambda,
            "var_nu": var_nu,
            "reward": reward,
            "ggf_value_xr": ggf_value_xr,
            "ggf_value_ln": ggf_value_ln,
        }
    )
    return results


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
