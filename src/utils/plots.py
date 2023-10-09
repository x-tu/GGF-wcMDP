"""This scripts include all plotting functions used to analyze results."""

import matplotlib.pyplot as plt


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
            plt.ylim(ymin=0, ymax=20)
            plt.title(f"Ep={ep}, Lr={lr}")
            plt.xlabel("Episodes")
            plt.ylabel("Discounted GGF Reward")
            sb_plot_index += 1
    plt.tight_layout()
    plt.show()
