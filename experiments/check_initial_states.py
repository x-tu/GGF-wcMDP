import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms.dlp import DLPAgent
from experiments.configs.base import params
from utils.encoding import state_int_index_to_vector

params.update({"num_episodes": 500, "len_episode": 100, "num_samples": 10})
timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
agent = DLPAgent(params=params)

all_rewards = {}
selected_states = list(
    range(0, params.num_states**params.num_groups + 1, params.num_groups - 1)
)
for state in selected_states:
    idx = state_int_index_to_vector(
        state_int_index=state, num_arms=params.num_groups, num_states=params.num_states
    )
    all_rewards[f"state {state}-{idx}"] = agent.run_mc_dlp(initial_state=state)
for key, ep_rewards in all_rewards.items():
    avg = np.cumsum(ep_rewards) / np.arange(1, len(ep_rewards) + 1)
    sns.lineplot(avg, label=key)
plt.ylim([16, 17])
plt.show()
pd.DataFrame(all_rewards).to_csv(f"results/dlp_mc_{timestamp}.csv")
