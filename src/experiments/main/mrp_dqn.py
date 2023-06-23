""" This script executes the DQN algorithm and its GGF counterpart for the machine replacement problem. """

import subprocess

# script path
script_path = "../configs/mrp_rl/"

# dqn (with stable baselines 3)
subprocess.run(["python", script_path + "dqn_mrp_sb3.py", "-ggi"])
subprocess.run(["python", script_path + "dqn_mrp_sb3.py"])
print("dqn done")
