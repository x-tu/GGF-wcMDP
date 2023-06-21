import argparse

from mrp_env import MachineReplace

from stable_baselines.common.vec_env import SubprocVecEnv


def make_env(rank, ggi):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """

    def _init():
        out_csv_name = (
            "results/reward_ppo_{}".format(rank)
            if not ggi
            else "results/reward_ppo_ggi_{}".format(rank)
        )
        env = MachineReplace(
            n_group=2, n_state=3, n_action=2, out_csv_name=out_csv_name, ggi=ggi
        )
        return env

    return _init


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""PPO"""
    )
    prs.add_argument(
        "-gam",
        dest="gamma",
        type=float,
        default=0.99,
        required=False,
        help="discount factor of PPO.\n",
    )
    prs.add_argument(
        "-a",
        dest="alpha",
        type=float,
        default=0.0005,
        required=False,
        help="Alpha learning rate.\n",
    )
    prs.add_argument(
        "-cr",
        dest="clip_range",
        type=float,
        default=0.2,
        required=False,
        help="clip_range of PPO.\n",
    )
    prs.add_argument(
        "-ggi", action="store_true", default=False, help="Run GGI algo or not.\n"
    )
    prs.add_argument(
        "-w",
        dest="weight",
        type=int,
        default=2,
        required=False,
        help="Weight coefficient\n",
    )
    args = prs.parse_args()

    # multiprocess environment
    n_cpu = 10
    ggi = args.ggi
    env = SubprocVecEnv([make_env(i, ggi) for i in range(n_cpu)])

    if ggi:
        from stable_baselines.common.policies_ggi import MlpPolicy as GGIMlpPolicy
        from stable_baselines.ppo2_ggi import PPO2_GGI

        model = PPO2_GGI(
            GGIMlpPolicy,
            env,
            gamma=args.gamma,
            verbose=0,
            reward_space=2,
            weight_coef=args.weight,
            learning_rate=args.alpha,
            cliprange=args.clip_range,
        )
    else:
        from stable_baselines import PPO2
        from stable_baselines.common.policies import MlpPolicy

        model = PPO2(
            MlpPolicy,
            env,
            gamma=args.gamma,
            verbose=0,
            learning_rate=args.alpha,
            cliprange=args.clip_range,
        )

    model.learn(total_timesteps=20000)
