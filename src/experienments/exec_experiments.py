""" This script executes the experiments for the machine replacement problem. """

import subprocess

# script path
script_path = "../env/machine_replacement/"

# run the experiment in parallel
# a2c
subprocess.run(["python", script_path + "a2c_mrp.py", "-ggi"])
subprocess.run(["python", script_path + "a2c_mrp.py"])
print("a2c done")

# # ppo
# subprocess.run(["python", script_path + "ppo_mrp.py", "-ggi"])
# subprocess.run(["python", script_path + "ppo_mrp.py"])
# print("ppo done")
#
# # dqn (with stable baselines 3)
# subprocess.run(["python", script_path + "dqn_mrp_sb3.py", "-ggi"])
# subprocess.run(["python", script_path + "dqn_mrp_sb3.py"])
# print("dqn done")
#
# # optimal agent
# for i in range(10):
#     subprocess.run(["python", script_path + "optimal_mrp.py", "-id", str(i)])
# print("optimal done")
#
# # random
# for i in range(10):
#     subprocess.run(["python", script_path + "random_mrp.py", "-id", str(i)])
# print("random done")
