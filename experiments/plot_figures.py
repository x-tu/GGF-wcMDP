import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plots import moving_average

data = {}
for algorithm in ["random", "sac", "ddpg", "ppo", "td3"]:
    df = pd.read_csv(f"experiments/tmp/{algorithm}.csv")["0"][:600]
    df = moving_average(df, window=10)
    label = algorithm.upper() if algorithm != "random" else "Random"
    data[label] = df
data_df = pd.DataFrame(data)
num_episodes = data_df.shape[0]

# Plot the data
data_df.plot()
# add whittle index
whittle_policy = pd.read_csv("experiments/tmp/rewards_whittle5.csv")
whittle_policy = whittle_policy.mean(axis=1)
wt_mean = whittle_policy.mean()
wt_std = whittle_policy.std() / np.sqrt(len(whittle_policy))
plt.axhline(y=wt_mean, color="r", linestyle="--", label="Whittle")
plt.fill_between(
    range(num_episodes), wt_mean - wt_std, wt_mean + wt_std, color="r", alpha=0.1
)

plt.ylim([8, 18])
plt.xlim([-5, 600])
plt.xlabel("Episodes")
plt.ylabel("Mean Expected Returns")
plt.title("Learning Curves (Smoothed)")
# show the legend
plt.legend()
plt.show()
