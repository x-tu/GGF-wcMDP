import matplotlib.pyplot as plt
import pandas as pd

from experiments.configs.base import params

group_rewards = pd.DataFrame()

for algorithm in ["PPO", "SAC", "TD3", "Whittle", "Random"]:
    df = pd.read_csv(
        f"experiments/tmp/rewards_{algorithm.lower()}_{params.identifier}.csv"
    )
    # drop the "Unnamed: 0" column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(labels="Unnamed: 0", axis=1)
    df.columns = [f"Machine {i+1}" for i in range(params.num_groups)]
    df["Algorithms"] = algorithm
    # concatenate the two dataframes
    group_rewards = pd.concat([group_rewards, df])

# plot the mean and variance of the two groups
bar_group_rewards = (
    group_rewards.groupby("Algorithms")
    .mean()
    .reindex(["PPO", "SAC", "TD3", "Whittle", "Random"])
)
bar_group_rewards.plot(
    kind="bar", yerr=group_rewards.groupby("Algorithms").std(), alpha=0.8
)

# label is horizontal
plt.xticks(rotation=0)
plt.ylim([0, 22])
plt.ylabel("Mean Discounted Returns")
plt.title(f"Mean Group Discounted Returns")
plt.savefig(f"experiments/tmp/bar_{params.identifier}.png")
plt.legend(loc="upper right")
plt.show()
