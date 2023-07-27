import argparse

from src.algorithms.tabular_q import run_tabular_Q
from src.env.mrp_env import MachineReplace
from src.experiments.configs.config_shared import prs_parser_setup

# configuration parameters
prs = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""Tabular Q Learning.""",
)
prs = prs_parser_setup(prs)
args = prs.parse_args()

ggi = False
out_csv_name = "../results/QL.csv" if not ggi else "../results/QL_ggi.csv"

env = MachineReplace(
    n_group=args.n_group,
    n_state=args.n_state,
    n_action=args.n_action,
    init_state=0,
    out_csv_name=out_csv_name,
    ggi=ggi,
)


policy = run_tabular_Q(
    env=env, num_episodes=200, len_episode=1000, alpha=0.005, epsilon=0.2, gamma=0.90
)

import numpy as np

# results for fixing the RL policy
from solver.fix_policy import (
    build_ggf_fix,
    extract_results as extract_results_fix,
    solve_ggf_fix,
)

# Build data
from src.utils.mrp_lp import MRPData

weight_coef = np.array([1 / (2 ** i) for i in range(args.n_group)])
data_mrp = MRPData(
    n_group=args.n_group,
    n_state=args.n_state,
    n_action=args.n_action,
    weight=weight_coef,
)  # TODO: not by default

model = build_ggf_fix(data_mrp, policy)
# Solve the GGF model
results, ggf_model = solve_ggf_fix(model=model)
extract_results_fix(model=ggf_model, data=data_mrp, policy_rl=policy)
