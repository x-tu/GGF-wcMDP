import argparse

import torch

# script path
script_path = "../env/machine_replacement/"

from src.env.mrp_env import MachineReplace
from src.experiments.configs.config_shared import prs_parser_setup
from stable_baselines3.common.fix_policy import policy_convertor_from_q_values
from stable_baselines3.dqn import DQN, MlpPolicy

# configuration parameters
prs = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""Deep Q Learning.""",
)
prs = prs_parser_setup(prs)
prs.add_argument(
    "-a",
    dest="alpha",
    type=float,
    default=0.001,
    required=False,
    help="Alpha learning rate.\n",
)
prs.add_argument(
    "-b",
    dest="buffer",
    type=int,
    default=5000,
    required=False,
    help="Buffer size of DQN.\n",
)
prs.add_argument(
    "-ef",
    dest="exploration_f",
    type=float,
    default=0.1,
    required=False,
    help="exploration_fraction.\n",
)
prs.add_argument(
    "-efs",
    dest="exploration_f_e",
    type=float,
    default=0.02,
    required=False,
    help="exploration_final_eps.\n",
)
prs.add_argument(
    "-batch",
    dest="batch_size",
    type=int,
    default=24,
    required=False,
    help="Batch size for NN.\n",
)
prs.add_argument(
    "-targfr",
    dest="target_net_freq",
    type=int,
    default=500,
    required=False,
    help="Update freq for target.\n",
)
args = prs.parse_args()

ggi = args.ggi
out_csv_name = (
    "../results/fix_policy_dqn" if not ggi else "../results/fix_policy_dqn_ggi"
)
env = MachineReplace(
    n_group=args.n_group,
    n_state=args.n_state,
    n_action=args.n_action,
    init_state=12,
    out_csv_name=out_csv_name,
    ggi=ggi,
)

reward_space = args.n_group

policy_kwargs = {
    "activation_fn": torch.nn.Tanh,
    "net_arch": [16, 16],
    "optimizer_kwargs": dict(weight_decay=0.001, eps=1e-5),
}

model = DQN(
    policy=MlpPolicy,
    env=env,
    learning_rate=args.alpha,
    buffer_size=args.buffer,
    exploration_fraction=args.exploration_f,
    exploration_final_eps=args.exploration_f_e,
    batch_size=args.batch_size,
    train_freq=50,
    gradient_steps=30,
    policy_kwargs=policy_kwargs,
    verbose=1,
    gamma=0.9,
    learning_starts=args.target_net_freq,
    target_update_interval=args.target_net_freq,
)

# learning of the model
trained_DQN = model.learn(total_timesteps=2000)
policy_rl = policy_convertor_from_q_values(trained_DQN)

import numpy as np

# Build data
from src.utils.mrp import MRPData

weight_coef = np.array([1 / (2 ** i) for i in range(reward_space)])
data_mrp = MRPData(
    n_group=args.n_group,
    n_state=args.n_state,
    n_action=args.n_action,
    weight=weight_coef,
)  # TODO: not by default

# results for ggf dual
from src.solver.ggf_dual import build_ggf, extract_results, solve_ggf

# Build t1he GGF model
ggf_model_lp = build_ggf(data=data_mrp)
# Solve the GGF model
results_lp, ggf_model_lp = solve_ggf(model=ggf_model_lp)
# Extract the results
extract_results(model=ggf_model_lp, data=data_mrp)

# results for fixing the RL policy
from solver.fix_policy import (
    build_ggf_fix,
    extract_results as extract_results_fix,
    solve_ggf_fix,
)

model = build_ggf_fix(data_mrp, policy_rl)
# Solve the GGF model
results, ggf_model = solve_ggf_fix(model=model)
extract_results_fix(model=ggf_model, data=data_mrp, policy_rl=policy_rl)
