import os
import subprocess
import sys

# script path
script_path = "./machine-ggf/"

# run the experiment
# a2c
subprocess.run(["python", script_path + "a2c_mrp.py", "-ggi"])
subprocess.run(["python", script_path + "a2c_mrp.py"])
print("a2c done")

# ppo
subprocess.run(["python", script_path + "ppo_mrp.py", "-ggi"])
subprocess.run(["python", script_path + "ppo_mrp.py"])
print("ppo done")

# dqn
subprocess.run(["python", script_path + "dqn_mrp.py", "-ggi"])
subprocess.run(["python", script_path + "dqn_mrp.py"])
print("dqn done")

# random
for i in range(10):
    subprocess.run(["python", script_path + "random_mrp.py", "-id", str(i)])
print("random done")
