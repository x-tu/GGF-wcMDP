"""This scripts include all plotting functions used to analyze results."""

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_figures(ep_list: list, lr_list: list, all_rewards: dict):
    """Plot the learning curves for different parameter settings.

    Args:
        ep_list (list): List of exploration rates.
        lr_list (list): List of learning rates.
        all_rewards (dict): Dictionary of rewards for different parameter settings.
    """

    plt.style.use("seaborn")
    plt.subplots(
        len(ep_list), len(lr_list), figsize=(4 * len(lr_list), 3 * len(ep_list))
    )
    # index for the subplots
    sb_plot_index = 1
    for ep in ep_list:
        for lr in lr_list:
            # Plot the learning curves for this parameter setting
            unique_key_str = f"{ep}-{lr}"
            rewards = all_rewards[unique_key_str]
            plt.subplot(len(ep_list), len(lr_list), sb_plot_index)
            plt.plot(rewards)
            plt.ylim(ymin=-20, ymax=0)
            plt.title(f"Ep={ep}, Lr={lr}")
            plt.xlabel("Episodes")
            plt.ylabel("Discounted GGF Reward")
            sb_plot_index += 1
    plt.tight_layout()
    plt.show()


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "episodes")
    y = moving_average(y, window=10)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    # plot a horizontal line showing the optimum
    plt.axhline(y=3.8275, color="r", linestyle="--")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Discounted Return")
    plt.title(title+ " Smoothed")
    plt.show()
