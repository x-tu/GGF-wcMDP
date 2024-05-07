import matplotlib.pyplot as plt
import pandas as pd

# concatenate the two dataframes
num_group = 5
group_rewards = pd.DataFrame()
for algorithm in ["PPO", "Whittle"]:
    df = pd.read_csv(f"experiments/tmp/rewards_{algorithm.lower()}{num_group}.csv")
    df.columns = [f"Group {i+1}" for i in range(num_group)]
    df["Algorithms"] = algorithm
    # concatenate the two dataframes
    group_rewards = pd.concat([group_rewards, df])

# plot the mean and variance of the two groups
group_rewards.groupby("Algorithms").mean().plot(
    kind="bar", yerr=group_rewards.groupby("Algorithms").std(), alpha=0.8
)

# label is horizontal
plt.xticks(rotation=0)
plt.ylim([0, 20])
plt.ylabel("Mean Discounted Returns")
plt.title(f"Mean Group Discounted Returns")
plt.show()
