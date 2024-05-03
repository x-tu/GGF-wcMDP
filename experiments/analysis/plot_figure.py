import os

import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3.common.results_plotter import load_results, ts2xy
from utils.plots import moving_average


def get_repo_root():
    current_dir = os.path.abspath(os.getcwd())

    while True:
        if os.path.isdir(os.path.join(current_dir, ".git")):
            return current_dir
        # Move up one directory
        current_dir = os.path.dirname(current_dir)
        # Stop if reached the root directory
        if current_dir == os.path.dirname(current_dir):
            break
    return None


repo_root = get_repo_root()

# get access to the parallel folder containing the CSV file
file_path = os.path.join(repo_root, "tmp", f"a2c.csv")
data = pd.read_csv(file_path)
data_np = data.values[:, 1:]
data_np = ((1 + 2 / 3) * 20 - data_np) / 2

# plot the data
mean = data_np.mean(axis=1)
std = data_np.std(axis=1)
# plt.plot(mean)
plt.plot(moving_average(data_np.mean(axis=1), window=3))
plt.fill_between(range(len(mean)), mean - std, mean + std, color="blue", alpha=0.1)
plt.axhline(y=3.8275, color="r", linestyle="--")
plt.xlabel("Episodes")
plt.ylabel("Mean Discounted Return")
plt.title(" Learning Curve (Smoothed)")
plt.show()
