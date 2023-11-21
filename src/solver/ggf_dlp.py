import random
from datetime import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from utils.common import MDP4LP, DotDict


def build_dlp(
    mdp: MDP4LP, deterministic_policy: bool = False, prob1_state_idx: int = None
) -> pyo.ConcreteModel:
    """Used to build the GGF dual MDP (stochastic) model."""

    model = pyo.ConcreteModel()
    model.deterministic_policy = deterministic_policy
    model.mdp = mdp

    # Create mu list (uniform by default)
    if isinstance(prob1_state_idx, int):
        big_mu_list = [0] * len(mdp.state_indices)
        big_mu_list[prob1_state_idx] = 1
        print("Initial distribution: ", big_mu_list)
    else:
        big_mu_list = [1 / len(mdp.state_indices)] * len(mdp.state_indices)
    model.init_distribution = big_mu_list

    # Variables
    model.varL = pyo.Var(mdp.group_indices, within=pyo.Reals)
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


def solve_dlp(model: pyo.ConcreteModel, num_opt_solutions: int = 1):
    """ Selects the solver and set the optimization settings.

    Args:
        model: the MRP model to be optimized
        num_opt_solutions: the number of optimal solutions to be returned

    Returns:
        results: the default optimization report
        model: the optimized model
        all_solutions: the list of all sub-solutions found

    """

    # Set the solver to be used
    start_time = datetime.now()
    opt = SolverFactory("gurobi_persistent", solver_io="python")
    opt.set_instance(model)

    if num_opt_solutions > 1:
        opt.set_gurobi_param("PoolSolutions", num_opt_solutions)
        opt.set_gurobi_param("PoolSearchMode", 2)

    # Solve the model
    results = opt.solve(model, tee=False, report_timing=False)
    print(
        f"Solver solving time: {round((datetime.now() - start_time).total_seconds(), 4)}"
    )

    num_opt_solutions = opt.get_model_attr("SolCount")
    print("Number of solutions found: " + str(num_opt_solutions))

    if num_opt_solutions > 1:
        # Print objective values of solutions
        all_solutions = []
        print("(Sub-)Optimal GGF Values: ")
        for e in range(num_opt_solutions):
            opt.set_gurobi_param("SolutionNumber", e)
            print("%g " % opt.get_model_attr("PoolObjVal"), end=" ")
            all_solutions.append(opt._solver_model.getAttr("Xn"))
        print("")
        all_solutions_dict = reformat_sub_solutions(
            all_solutions=all_solutions, model=model
        )
        return results, model, all_solutions_dict
    return results, model, None


def extract_dlp(model: pyo.ConcreteModel, print_results: bool = False):
    """ This function is used to extract optimized results.

    Args:
        model: the MRP model to be optimized
        print_results: whether to print the results

    Returns:
        results: the extracted results

    """
    # extract the policy Pi and visitation frequency X
    policy_np = np.zeros((model.mdp.num_states, model.mdp.num_actions))
    var_x_np = np.zeros((model.mdp.num_states, model.mdp.num_actions))
    for s in model.mdp.state_indices:
        x_sum = (
            sum([model.varX[s, a].value for a in model.mdp.action_indices])
            if not model.deterministic_policy
            else None
        )
        for a in model.mdp.action_indices:
            var_x_np[s, a] = model.varX[s, a].value
            if model.deterministic_policy:
                if 0 < model.varPi[s, a].value < 1:
                    policy_np[s, a] = round(model.varPi[s, a].value, 4)
                else:
                    policy_np[s, a] = round(model.varPi[s, a].value)
            else:
                policy_np[s, a] = model.varX[s, a].value / max(
                    x_sum, 1e-6
                )  # avoid zero division
    # convert to dataframe
    s_idx = np.array(
        [
            str(model.mdp.state_tuple_list[s]) if not model.mdp.encoding_int else str(s)
            for s in model.mdp.state_indices
        ]
    )
    var_x = pd.DataFrame(var_x_np, index=s_idx, columns=model.mdp.action_indices).round(
        4
    )
    policy = pd.DataFrame(
        policy_np, index=s_idx, columns=model.mdp.action_indices
    ).round(4)

    # extract the dual variables L, N
    dual_var_df = pd.DataFrame(
        index=model.mdp.group_indices, columns=["Var L", "Var N"]
    )
    for d in model.mdp.group_indices:
        dual_var_df.loc[d] = [
            round(model.varL[d].value, 4),
            round(model.varN[d].value, 4),
        ]

    # Count the proportion of deterministic policy to positive policy
    proportion = np.sum((policy_np > 0) & (policy_np < 1)) / np.sum(policy_np > 0)
    # Calculate the cost for each group
    costs = calculate_group_cost(model=model)
    # GGF value
    cost_sorted = (
        np.sort(costs)[::-1]
        if model.objective.sense == pyo.minimize
        else np.sort(costs)
    )
    ggf_value_xc = np.dot(cost_sorted, model.mdp.weights).round(4)

    results = DotDict(
        {
            "var_x": var_x,
            "policy": policy,
            "var_dual": dual_var_df,
            "costs": costs,
            "ggf_value_xc": ggf_value_xc,
            "ggf_value_ln": round(
                sum(dual_var_df["Var L"]) + sum(dual_var_df["Var N"]), 4
            ),
        }
    )

    if print_results:
        print(f"Proportion of stochastic policy: {round(proportion * 100, 2)}%")
        format_prints(results=results, model=model)
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


def calculate_group_cost(model: pyo.ConcreteModel) -> np.array:
    # Costs for each group
    costs = []
    for d in model.mdp.group_indices:
        all_cost = round(
            sum(
                model.mdp.costs[s, a, d] * model.varX[s, a].value
                for s in model.mdp.state_indices
                for a in model.mdp.action_indices
            ),
            4,
        )
        costs.append(all_cost)
    return np.array(costs)


def format_prints(results: DotDict, model: pyo.ConcreteModel):
    """ This function is used to format the results and print them.

    Args:
        results: the results to be printed
        model: the DLP model

    """
    # disable the output limit
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)

    cost_g1_df = pd.DataFrame(
        model.mdp.costs[:, :, 0],
        index=results.policy.index,
        columns=model.mdp.action_indices,
    ).round(2)
    cost_g2_df = pd.DataFrame(
        model.mdp.costs[:, :, 1],
        index=results.policy.index,
        columns=model.mdp.action_indices,
    ).round(2)

    policy_formatted = results.policy.apply(
        lambda x: x.map(lambda val: round(val, 2) if 0 < val < 1 else str(int(val)))
    )
    var_x_formatted = results.var_x.apply(
        lambda x: x.map(lambda val: str(int(val)) if val == 0 else round(val, 4))
    )
    space_df = pd.DataFrame(
        [" "] * model.mdp.num_states, index=results.policy.index, columns=[" "]
    )
    concat_df = pd.concat(
        [
            policy_formatted,
            space_df,
            var_x_formatted,
            space_df,
            space_df,
            cost_g1_df,
            space_df,
            cost_g2_df,
        ],
        axis=1,
    )
    space_size = 12 + model.mdp.num_actions * 4
    print(
        f"Policy:{' ' * space_size}Var X:{' ' * int(space_size/2)}Costs - Group 1 | Group 2{' ' * space_size}\n{concat_df}"
    )

    space_df = pd.DataFrame(
        [" "] * model.mdp.num_groups, index=model.mdp.group_indices, columns=[" "]
    )
    cost_df = pd.DataFrame(
        results.costs, index=model.mdp.group_indices, columns=["Group Costs"]
    )
    print(pd.concat([results.var_dual, space_df, cost_df], axis=1))

    print("Var X total:", sum(results.var_x.sum()))
    print("GGF Value (DLP) L+N: ", results.ggf_value_ln)
    print("GGF Value (DLP) XC:  ", results.ggf_value_xc)


def reformat_sub_solutions(all_solutions: list, model: pyo.ConcreteModel):
    num_groups = len(model.mdp.group_indices)
    all_results = {}
    for sol_idx in range(len(all_solutions)):
        sol = all_solutions[sol_idx]
        varL = np.array(sol[0:num_groups])
        varN = np.array(sol[num_groups : 2 * num_groups])
        dual_var_df = pd.DataFrame(
            np.zeros((model.mdp.num_groups, 2)),
            index=model.mdp.group_indices,
            columns=["Var L", "Var N"],
        ).round(4)
        for d in model.mdp.group_indices:
            dual_var_df.loc[d, "Var L"] = round(varL[d], 4)
            dual_var_df.loc[d, "Var N"] = round(varN[d], 4)

        if model.deterministic_policy:
            policy_np = (
                np.array(
                    sol[
                        2 * num_groups : 2 * num_groups
                        + model.mdp.num_states * model.mdp.num_actions
                    ]
                )
                .round(1)
                .reshape(model.mdp.num_states, model.mdp.num_actions)
            )
            var_x_np = (
                np.array(
                    sol[
                        2 * num_groups
                        + model.mdp.num_states
                        * model.mdp.num_actions : 2
                        * model.mdp.num_groups
                        + 2 * model.mdp.num_states * model.mdp.num_actions
                    ]
                )
                .round(4)
                .reshape(model.mdp.num_states, model.mdp.num_actions)
            )
        else:
            var_x_np = (
                np.array(
                    sol[
                        2 * num_groups : 2 * num_groups
                        + model.mdp.num_states * model.mdp.num_actions
                    ]
                )
                .round(1)
                .reshape(model.mdp.num_states, model.mdp.num_actions)
            )
            policy_np = np.zeros((model.mdp.num_states, model.mdp.num_actions))
            for s in model.mdp.state_indices:
                x_sum = sum([var_x_np[s, a] for a in model.mdp.action_indices])
                for a in model.mdp.action_indices:
                    policy_np[s, a] = var_x_np[s, a] / max(x_sum, 1e-6)
        # convert to dataframe
        s_idx = np.array(
            [
                str(model.mdp.state_tuple_list[s])
                if not model.mdp.encoding_int
                else str(s)
                for s in model.mdp.state_indices
            ]
        )
        var_x = pd.DataFrame(
            var_x_np, index=s_idx, columns=model.mdp.action_indices
        ).round(4)
        policy = pd.DataFrame(
            policy_np, index=s_idx, columns=model.mdp.action_indices
        ).round()
        costs = calculate_group_cost(model=model)
        costs_sorted = (
            np.sort(costs)[::-1]
            if model.objective.sense == pyo.minimize
            else np.sort(costs)
        )
        ggf_value_xc = np.dot(costs_sorted, model.mdp.weights).round(4)

        all_results[sol_idx] = DotDict(
            {
                "var_x": var_x,
                "policy": policy,
                "var_dual": dual_var_df,
                "costs": costs,
                "ggf_value_xc": ggf_value_xc,
                "ggf_value_ln": round(
                    sum(dual_var_df["Var L"]) + sum(dual_var_df["Var N"]), 4
                ),
            }
        )
    return all_results
