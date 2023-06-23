import argparse

from src.env.mrp_env import MachineReplace

if __name__ == "__main__":
    ggi = False
    env = MachineReplace(
        n_group=2,
        n_state=3,
        n_action=2,
        out_csv_name="../results/reward_random",
        ggi=ggi,
    )
    obs = env.reset()
    for i in range(20000):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            env.reset()
