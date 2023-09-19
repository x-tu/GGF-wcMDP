""" This script executes the RL algorithms and their GGF counterpart for the machine replacement problem. """

import subprocess

# script path
script_path = "../configs/mrp_rl/"

# run the experiment in parallel
# a2c
subprocess.run(["python", script_path + "a2c_mrp.py", "-ggi"])
subprocess.run(["python", script_path + "a2c_mrp.py"])
print("a2c done")

# ppo
# subprocess.run(["python", script_path + "ppo_mrp.py", "-ggi"])
# subprocess.run(["python", script_path + "ppo_mrp.py"])
# print("ppo done")

# dqn
# subprocess.run(["python", script_path + "dqn_mrp.py", "-ggi"])
# subprocess.run(["python", script_path + "dqn_mrp.py"])
# print("dqn done")

# optimal agent
# for i in range(10):
#     subprocess.run(["python", script_path + "optimal_mrp.py", "-id", str(i)])
# print("optimal done")

# random
# for i in range(10):
#     subprocess.run(["python", script_path + "random_mrp.py"])
# print("random done")
