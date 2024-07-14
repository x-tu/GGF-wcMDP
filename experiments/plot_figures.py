import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.configs.base import params
from utils.plots import moving_average

data = {}
for algorithm in ["ppo", "sac", "td3"]:
    df = pd.read_csv(
        f"experiments/tmp/learning_reward_{algorithm}_{params.identifier}.csv"
    )["0"]
    df = moving_average(df, window=10)
    label = algorithm.upper()
    data[label] = df
data_df = pd.DataFrame(data)
num_episodes = int(np.ceil(data_df.shape[0] / 100) * 100)

weights = np.array([1 / (2**i) for i in range(params.num_groups)])
weights /= np.sum(weights)

# random
df = pd.read_csv(f"experiments/tmp/rewards_random_{params.identifier}.csv")
# drop the first column
df = df.drop(df.columns[0], axis=1)
df_random = moving_average(
    np.dot(np.sort(df), params.weights).round(params.digit), window=10
)
data_df["Random"] = df_random[0 : data_df.shape[0]]

# Plot the data
data_df.plot()
# add whittle index
whittle_policy = pd.read_csv(f"experiments/tmp/rewards_whittle_{params.identifier}.csv")
whittle_policy = whittle_policy.mean(axis=1)
wt_mean = whittle_policy.mean()
wt_std = whittle_policy.std() / np.sqrt(len(whittle_policy))
plt.axhline(y=wt_mean, color="r", linestyle="--", label="Whittle")
plt.fill_between(
    range(num_episodes), wt_mean - wt_std, wt_mean + wt_std, color="r", alpha=0.1
)
ymax = np.ceil(data_df.max().max()) + 1

plt.ylim([8, ymax])
plt.xlim([-5, num_episodes])
plt.xlabel("Episodes")
plt.ylabel("GGF Expected Returns")
plt.title("Learning Curves (Smoothed)")
# set figure size
plt.gcf().set_size_inches(6, 5)
plt.savefig(f"experiments/tmp/{params.identifier}.png")
plt.legend()
plt.show()
