import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.plots import moving_average

random = pd.read_csv("experiments/tmp/random_2.csv")["0"]
a2c = pd.read_csv("experiments/tmp/a2c-new.csv")["0"]
ddpg = pd.read_csv("experiments/tmp/ddpg-new.csv")["0"]

random_mv = moving_average(random, window=10)
a2c_mv = moving_average(a2c, window=10)
ddpg_mv = moving_average(ddpg, window=10)

data = pd.DataFrame({"Random": random_mv, "A2C": a2c_mv, "DDPG": ddpg_mv})

# Plot the data
data.plot()
plt.xlabel("Episodes")
plt.ylabel("Mean Expected Returns")
plt.title("Learning Curves (Smoothed)")
plt.show()
