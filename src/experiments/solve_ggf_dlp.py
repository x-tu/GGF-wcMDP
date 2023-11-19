from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, solve_dlp
from utils.common import MDP4LP
from utils.mrp import MRPData

for group in [2, 3, 4]:
    print(f"\n{'='*20} Group: {group} {'='*20}")
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
            # weight_type="uniform",
            # cost_types_operation=["quadratic"] * params.num_groups,
            # cost_types_replace=["quadratic"] * params.num_groups,
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
            mdp=mdp, deterministic_policy=deterministic, prob1_state_idx=None
        )
        _, model, all_solutions = solve_dlp(model=model, num_opt_solutions=1)
        results = extract_dlp(model=model, print_results=True)
