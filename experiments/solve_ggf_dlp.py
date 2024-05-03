import numpy as np
import pandas as pd
from termcolor import colored

from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, format_prints, solve_dlp
from utils.common import MDP4LP
from utils.encoding import state_int_index_to_vector
from utils.mrp import MRPData
from utils.policy import calculate_state_value, calculate_visitation_freq

print_results = {
    "print_solve_results": True,
    "print_var_x_decomposed": False,
    "print_var_x_recalculation": False,
    "print_all_sub_solutions": False,
    "print_constraint_check": False,
}

weight_types, cost_type_operations, cost_type_replaces = (
    ["exponential3"],  # ["uniform", "exponential2", "exponential3", "random"],
    ["linear"],  # ["constant","linear","quadratic","exponential","rccc","random",]
    ["zero"],  # ["zero","constant","linear","quadratic","exponential","rccc","random",]
)

params.update(
    {
        "num_states": 2,
        "prob_remain": 0.0,
        "num_opt_solutions": 1,
        "prob1_state_idx": None,
    }
)


def main(params):
    for group in [2]:
        print(f"\n{'=' * 20} Group: {group} {'=' * 30}")
        gaps = {}
        for weight_type in weight_types:
            for cost_type_operation in cost_type_operations:
                for cost_type_replace in cost_type_replaces:
                    ggf = []
                    params.seed += 1
                    for deterministic in [False, True]:
                        params.update(
                            {
                                "num_groups": group,
                                "deterministic_policy": deterministic,
                                "weight_type": weight_type,
                                "cost_type_replace": cost_type_replace,
                                "cost_type_operation": cost_type_operation,
                            }
                        )
                        # solve the DLP
                        results, mdp, mrp_data, model, all_solutions = solve_dlp_once(
                            params
                        )
                        ggf.append(results.ggf_value_ln)

                        # check the results
                        if print_results["print_var_x_decomposed"]:
                            print_decomposed_x(var_x=results.var_x, costs=mdp.costs)
                        if print_results["print_var_x_recalculation"]:
                            print_recalculated_x(mdp, model, results)
                        if print_results["print_all_sub_solutions"] and all_solutions:
                            for key, sub_results in all_solutions.items():
                                print(colored(f"\n>>> >>> Solution {key + 1}:", "blue"))
                                format_prints(results=sub_results, model=model)
                        if print_results["print_constraint_check"]:
                            for i in range(mrp_data.num_groups):
                                for j in range(mrp_data.num_groups):
                                    print(
                                        f"{round(results.var_dual['Var L'][i] + results.var_dual['Var N'][j], 4)} "
                                        f">= {round(mrp_data.weights[i] * results.costs[j], 4)}  "
                                        f"(L{i} + N{j} >= w{i} * Xr{j})"
                                    )
                    gaps[f"{weight_type}-{cost_type_operation}-{cost_type_replace}"] = {
                        "(D-S)/D%": round(((ggf[-1] - ggf[0]) / ggf[-1]) * 100, 2),
                        "S": round(ggf[0], 4),
                        "D": round(ggf[-1], 4),
                    }
                    if len(ggf) == 2:
                        print(
                            f"\nGap ({weight_type}-{cost_type_operation}-{cost_type_replace}): "
                            f"{round(((ggf[-1] - ggf[0]) / ggf[-1]) * 100, 2)}%"
                        )
                    # assertion
                    assert (
                        abs(results.ggf_value_xc - results.ggf_value_ln) < 1e-3
                    ), f"{results.ggf_value_xc} != {results.ggf_value_ln}"
        # save the gaps
        gaps_df = pd.DataFrame.from_dict(gaps, orient="index")
        gaps_df.to_csv(f"results/gaps_group{group}_state{params.num_states}.csv")


def solve_dlp_once(params):
    POLICY_STR = "deterministic" if params.deterministic_policy else "stochastic"
    HEADER_STR = (
        f"\n>>> Policy: {POLICY_STR} | "
        f"{params.num_groups} groups, {params.num_states} states, {params.num_actions} actions "
        f"({params.weight_type}-{params.cost_type_operation}-{params.cost_type_replace})"
    )
    print(colored(HEADER_STR, "red"))
    mrp_data = MRPData(
        num_groups=params.num_groups,
        num_states=params.num_states,
        num_actions=params.num_actions,
        prob_remain=params.prob_remain,
        add_absorbing_state=False,
        weight_type=params.weight_type,
        cost_types_operation=params.cost_type_operation,
        cost_types_replace=params.cost_type_replace,
        seed=params.seed,
    )

    mdp = MDP4LP(
        num_states=mrp_data.num_global_states,
        num_actions=mrp_data.num_global_actions,
        num_groups=mrp_data.num_groups,
        transition=mrp_data.global_transitions,
        costs=mrp_data.global_costs,
        discount=params.gamma,
        weights=mrp_data.weights,
        minimize=True,
        encoding_int=False,
        base_num_states=params.num_states,
    )
    model = build_dlp(
        mdp=mdp,
        deterministic_policy=params.deterministic_policy,
        prob1_state_idx=params.prob1_state_idx,
    )
    _, model, all_solutions = solve_dlp(
        model=model, num_opt_solutions=params.num_opt_solutions
    )
    results = extract_dlp(
        model=model, print_results=print_results["print_solve_results"]
    )
    # save the policy
    results.policy.to_csv(
        f"results/policy/{POLICY_STR}_policy_group{params.num_groups}_state{params.num_states}.csv"
    )
    return results, mdp, mrp_data, model, all_solutions


def print_recalculated_x(mdp, model, results):
    visitation_freq = calculate_visitation_freq(
        discount=mdp.discount,
        initial_state_prob=model.init_distribution,
        policy=results.policy.to_numpy(),
        transition_prob=mdp.transition,
        time_horizon=200,
    )
    varX_recalculation = results.var_x.copy()
    varX_recalculation[:] = visitation_freq.round(4)

    value_list = calculate_state_value(
        discount=mdp.discount,
        initial_state_prob=model.init_distribution,
        policy=results.policy.to_numpy(),
        reward_or_cost=mdp.costs,
        transition_prob=mdp.transition,
        time_horizon=200,
    )
    cost_sorted = np.sort(np.array(value_list[-1]))
    ggf_value_xc = round(np.dot(cost_sorted, model.mdp.weights), 4)
    space_df = pd.DataFrame(
        [" "] * model.mdp.num_states, index=results.var_x.index, columns=[" "]
    )
    format_rule = lambda x: x.map(
        lambda val: str(int(val)) if val == 0 else round(val, 4)
    )
    concat_df = pd.concat(
        [
            varX_recalculation.apply(format_rule),
            space_df,
            results.var_x.apply(format_rule),
        ],
        axis=1,
    )
    space_size = 4 + model.mdp.num_actions * 4
    print("GGF Value (Re-calculation) XC:", ggf_value_xc)
    print(f"Var X (Re-calculation):{' ' * space_size}Var X (DLP):\n{concat_df}")


def print_decomposed_x(var_x, costs):
    S, A, D = costs.shape
    # get the D root of S
    num_states = int(S ** (1 / D))
    var_x_by_group = np.zeros((num_states, A, D))
    costs_by_group = np.zeros((num_states, A, D))
    for s in range(S):
        s_vec = state_int_index_to_vector(s, D, num_states)
        var_x_by_state = var_x.iloc[s]
        costs_by_state = costs[s, :, :]
        for d in range(D):
            var_x_by_group[s_vec[d], :, d] += var_x_by_state
            costs_by_group[s_vec[d], :, d] += costs_by_state[:, d]
    costs_by_group = costs_by_group / num_states
    costs_by_group = costs_by_group.round(2)
    # covert and concate to dataframe
    space_df = pd.DataFrame(
        [" "] * num_states, index=np.arange(num_states), columns=[" "]
    )
    for d in range(D):
        print(f"\n>>> Group {d} (Var X | Cost)")
        results_df = pd.DataFrame(index=np.arange(num_states))
        var_x_df = pd.DataFrame(
            var_x_by_group[:, :, d], index=np.arange(num_states), columns=var_x.columns
        )
        cost_df = pd.DataFrame(
            costs_by_group[:, :, d], index=np.arange(num_states), columns=var_x.columns
        )
        results_df = pd.concat([results_df, var_x_df, space_df], axis=1)
        results_df = pd.concat([results_df, cost_df], axis=1)
        print(results_df)


main(params)
