import numpy as np
import pandas as pd

from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, format_prints, solve_dlp
from utils.common import MDP4LP
from utils.mrp import MRPData
from utils.policy import calculate_state_value, calculate_visitation_freq

print_results = {
    "print_solve_results": False,
    "print_var_x_recalculation": False,
    "print_all_sub_solutions": False,
}

for group in [4]:
    print(f"\n{'='*20} Group: {group} {'='*30}")
    gaps = {}
    for weight_type in ["uniform", "exponential2", "exponential3", "random"]:
        for cost_type_operation in [
            "constant",
            "linear",
            "quadratic",
            "exponential",
            "rccc",
            "random",
        ]:
            for cost_type_replace in [
                "zero",
                "constant",
                "linear",
                "quadratic",
                "exponential",
                "rccc",
                "random",
            ]:
                ggf = []
                for deterministic in [False, True]:
                    POLICY_STR = "deterministic" if deterministic else "stochastic"
                    print(f"\n>>> Policy: {POLICY_STR}")
                    params.update({"num_groups": group, "num_states": 3})
                    mrp_data = MRPData(
                        num_groups=params.num_groups,
                        num_states=params.num_states,
                        num_actions=params.num_actions,
                        prob_remain=params.prob_remain,
                        add_absorbing_state=False,
                        weight_type=weight_type,
                        cost_types_operation=[cost_type_operation] * params.num_groups,
                        cost_types_replace=[cost_type_replace] * params.num_groups,
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
                        deterministic_policy=deterministic,
                        prob1_state_idx=None,
                    )
                    _, model, all_solutions = solve_dlp(
                        model=model, num_opt_solutions=1
                    )
                    results = extract_dlp(
                        model=model, print_results=print_results["print_solve_results"]
                    )
                    ggf.append(results.ggf_value_ln)
                    # save the policy
                    results.policy.to_csv(
                        f"results/policy/{POLICY_STR}_policy_group{group}_state{params.num_states}.csv"
                    )

                    if print_results["print_var_x_recalculation"]:
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
                            rewards=mdp.costs,
                            transition_prob=mdp.transition,
                            time_horizon=200,
                        )
                        reward_sorted = np.sort(np.array(value_list[-1]))
                        ggf_value_xr = round(
                            np.dot(reward_sorted, model.mdp.weights), 4
                        )
                        space_df = pd.DataFrame(
                            [" "] * model.mdp.num_states,
                            index=results.var_x.index,
                            columns=[" "],
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
                        print("GGF Value (Re-calculation) XR:", ggf_value_xr)
                        print(
                            f"Var X (Re-calculation):{' ' * space_size}Var X (DLP):\n{concat_df}"
                        )

                    if print_results["print_all_sub_solutions"] and all_solutions:
                        for key, results in all_solutions.items():
                            print(f"\n>>> >>> Solution {key+1}:")
                            format_prints(results=results, model=model)
                print(
                    f"\nGap ({weight_type}-{cost_type_operation}-{cost_type_replace}): {round(((ggf[-1] - ggf[0]) / ggf[0]) * 100, 2)}%"
                )
                gaps[f"{weight_type}-{cost_type_operation}-{cost_type_replace}"] = {
                    "(D-S)/S": round(((ggf[-1] - ggf[0]) / ggf[0]) * 100, 2),
                    "S": round(ggf[0], 4),
                    "D": round(ggf[-1], 4),
                }
    # save the gaps
    gaps_df = pd.DataFrame.from_dict(gaps, orient="index")
    gaps_df.to_csv(f"results/gaps_group{group}_state{params.num_states}.csv")
