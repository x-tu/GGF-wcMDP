import json

from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, solve_dlp
from utils.common import MDP4LP
from utils.mrp import MRPData

records = {}

for group in [2, 3, 4, 5, 6, 7]:
    print(f"\n=== Group: {group} ===")
    for deterministic in [False, True]:
        POLICY_STR = "deterministic" if deterministic else "stochastic"
        print(f">>> Policy: {POLICY_STR}")

        params.update({"num_groups": group})

        mrp_data = MRPData(
            num_groups=params.num_groups,
            num_states=params.num_states,
            num_actions=params.num_actions,
            prob_remain=params.prob_remain,
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
        model = build_dlp(mdp=mdp, deterministic_policy=deterministic)
        _, model = solve_dlp(model)
        results = extract_dlp(model)

        records[f"{group}_{deterministic}"] = results

with open("results/ggf_dlp.json", "w") as f:
    json.dump(records, f)
