import argparse

import numpy as np
from src.env.mrp_env import MachineReplace


def get_optimal_action(state):
    if state in [0, 1, 3, 4]:
        return 0
    elif state in [6, 7]:
        return 1
    elif state in [2, 5]:
        return 2
    else:
        return np.random.choice([1, 2])


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Optimal MRP Agent""",
    )
    prs.add_argument(
        "-fr",
        dest="ifr",
        type=int,
        default=2,
        required=False,
        help="Functional Response for SC\n",
    )
    prs.add_argument(
        "-fnum",
        dest="ifrnum",
        type=int,
        default=2,
        required=False,
        help="Functional Response Num for SC\n",
    )
    prs.add_argument("-id", dest="run_index", type=int, default=0, help="Run index.\n")
    args = prs.parse_args()

    ggi = False
    env = MachineReplace(
        n_group=2,
        n_state=3,
        n_action=2,
        out_csv_name=f"../results/reward_optimal_{args.run_index}",
        ggi=ggi,
    )
    obs = env.reset()
    for i in range(2001):
        action = get_optimal_action(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            env.reset()
