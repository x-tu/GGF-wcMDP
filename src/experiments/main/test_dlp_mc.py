import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from algorithms.dlp import DLPAgent
from experiments.configs.base import params

params.update({"num_episodes": 500, "len_episode": 100, "num_samples": 10})
timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
ep_rewards = DLPAgent(params=params).run_mc_dlp()
sns.lineplot(ep_rewards)
plt.show()
