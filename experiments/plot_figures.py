import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, solve_dlp
from utils.common import MDP4LP

# solve the optimal
from utils.mrp import MRPData
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
plt.axhline(y=wt_mean, color="purple", linestyle="--", label="Whittle")
plt.fill_between(
    range(num_episodes), wt_mean - wt_std, wt_mean + wt_std, color="r", alpha=0.1
)
ymax = np.ceil(data_df.max().max()) + 1

mrp_data = MRPData(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    prob_remain=params.prob_remain,
    weight_type=params.weight_type,
    force_to_use_all_resources=params.force_to_use_all_resources,
    cost_types_operation=params.cost_type_operation,
    cost_types_replace=params.cost_type_replace,
)
mdp = MDP4LP(
    num_states=mrp_data.num_global_states,
    num_actions=mrp_data.num_global_actions,
    num_groups=mrp_data.num_groups,
    transition=mrp_data.global_transitions,
    costs=mrp_data.global_costs,
    discount=params.gamma,
    weights=mrp_data.weights,
    minimize=True,
    encoding_int=False,
    base_num_states=params.num_states,
)
# calculate LP values
uniform_dist = [
    1 / mrp_data.num_global_states for i in range(mrp_data.num_global_states)
]
model = build_dlp(mdp=mdp, initial_mu=uniform_dist)
# Solve the GGF model
_, model, _ = solve_dlp(model=model)
results = extract_dlp(model=model, print_results=False)
optimal_value = 20 - results.ggf_value_xc
print("Optimal Value: ", optimal_value)
plt.axhline(y=optimal_value, color="r", linestyle="--", label="Optimal")

plt.ylim([8, ymax])
plt.xlim([-5, num_episodes])
plt.xlabel("Episodes")
plt.ylabel("GGF Expected Returns")
plt.title("Learning Curves (Smoothed)")
# set figure size
plt.gcf().set_size_inches(6, 5)
plt.legend()
plt.savefig(f"experiments/tmp/{params.identifier}.png")
plt.show()
