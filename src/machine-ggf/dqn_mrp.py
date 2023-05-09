import argparse

from mrp_env import MachineReplace

from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.deepq_ggi import DQN_GGI, MlpPolicy as GGIMlpPolicy

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Deep-Q-Learning""",
    )
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
        default=50000,
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
        default=64,
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
    prs.add_argument(
        "-w",
        dest="weight",
        type=int,
        default=3,
        required=False,
        help="Weight coefficient\n",
    )
    prs.add_argument(
        "-ggi", action="store_true", default=False, help="Run GGI algo or not.\n"
    )
    args = prs.parse_args()
    ggi = args.ggi
    out_csv_name = "results/reward_dqn" if not ggi else "results/reward_dqn_ggi"
    env = MachineReplace(
        n_group=2, n_state=3, n_action=2, out_csv_name=out_csv_name, ggi=ggi
    )
    weight_coef = args.weight

    if ggi:
        model = DQN_GGI(
            env=env,
            policy=GGIMlpPolicy,
            weight_coef=args.weight,
            learning_rate=args.alpha,
            buffer_size=args.buffer,
            exploration_fraction=args.exploration_f,
            exploration_final_eps=args.exploration_f_e,
            batch_size=args.batch_size,
            double_q=False,
            target_network_update_freq=args.target_net_freq,
        )
    else:
        model = DQN(
            env=env,
            policy=MlpPolicy,
            learning_rate=args.alpha,
            buffer_size=args.buffer,
            exploration_fraction=args.exploration_f,
            exploration_final_eps=args.exploration_f_e,
            batch_size=args.batch_size,
            double_q=False,
            target_network_update_freq=args.target_net_freq,
        )
    # learning of the model
    model.learn(total_timesteps=20000)
