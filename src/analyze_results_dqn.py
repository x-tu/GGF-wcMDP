import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

save_bbox = Bbox([[0.7, 0.2], [9.2, 4.5]])


def read_csv(fnPattern):
    # find files with fileformat
    pattern = re.compile(fnPattern + "_[0-9]+.csv")
    files = [f for f in os.listdir("results") if pattern.match(f)]
    dfs = []
    for f in files:
        tmp_df = pd.read_csv("results/" + f, names=["reward_0", "reward_1"])
        tmp_df["step"] = tmp_df.index
        # if terminal_observation exists, drop it
        if "terminal_observation" in tmp_df.columns:
            tmp_df = tmp_df.drop(columns=["terminal_observation"])
        dfs.append(tmp_df)
    # concat all dataframes
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by=["step"], inplace=True, ignore_index=True)
    return df


def compute_GGF(df, reward_n=2, weight_coef=2):
    # omega = np.array([1 / (weight_coef ** i) for i in range(reward_n)])
    omega = np.array([0.67, 0.33])
    max_val = df[["reward_0", "reward_1"]].max(axis=1)
    min_val = df[["reward_0", "reward_1"]].min(axis=1)
    # compute GGF, dot product of omega and two values
    GGF = np.dot(omega, np.array([min_val, max_val]))
    return GGF


def read_result_data(pattern_list, keys_list=None):
    if keys_list is None:
        keys_list = pattern_list
    df_list = []
    for pattern, key in zip(pattern_list, keys_list):
        tmp_df = read_csv(pattern)
        tmp_df["GGF_Score"] = compute_GGF(tmp_df, reward_n=2, weight_coef=2)
        # subset of df_ppo
        # tmp_df = tmp_df[tmp_df['step']%20 == 18]
        tmp_df = tmp_df[tmp_df["step"] < 100000]
        tmp_df["Algorithm"] = [key] * tmp_df.shape[0]
        df_list.append(tmp_df)
    # merge dataframes
    data_ = pd.concat(df_list, ignore_index=True)
    data_["Sum"] = data_["reward_0"] + data_["reward_1"]
    return data_


def plot_sum_cost(df, algs_list):
    plt.figure(figsize=(10, 5))
    for alg in algs_list:
        alg_df = df[df["Algorithm"] == alg]
        group_df = alg_df.groupby(["step"]).first().reset_index()[["step"]]
        group_df["Sum_mean"] = (
            alg_df.groupby(["step"])["Sum"].mean().reset_index()["Sum"]
        )
        group_df["Sum_std"] = alg_df.groupby(["step"])["Sum"].std().reset_index()["Sum"]
        # make the Sum_mean smoother and keep the same length
        ma_length = 20
        group_df["Sum_mean"] = (
            group_df["Sum_mean"].rolling(ma_length, min_periods=1).mean()
        )
        group_df["Sum_std"] = (
            group_df["Sum_std"].rolling(ma_length, min_periods=1).mean()
        )

        plt.plot(group_df["step"], group_df["Sum_mean"], label=alg, alpha=0.8)
        plt.fill_between(
            group_df["step"],
            group_df["Sum_mean"] - group_df["Sum_std"],
            group_df["Sum_mean"] + group_df["Sum_std"],
            alpha=0.2,
        )

    plt.xlabel("Number of Steps")
    plt.ylabel("Total costs")
    plt.legend(ncol=4, loc="lower right")
    plt.show()


def plot_ggf(df, algs_list):
    plt.figure(figsize=(10, 5))
    for alg in algs_list:
        alg_df = df[df["Algorithm"] == alg]
        group_df = alg_df.groupby(["step"]).first().reset_index()[["step"]]
        group_df["GGF_mean"] = (
            alg_df.groupby(["step"])["GGF_Score"].mean().reset_index()["GGF_Score"]
        )
        group_df["GGF_std"] = (
            alg_df.groupby(["step"])["GGF_Score"].std().reset_index()["GGF_Score"]
        )
        # make the Sum_mean smoother and keep the same length
        ma_length = 20
        group_df["GGF_mean"] = (
            group_df["GGF_mean"].rolling(ma_length, min_periods=1).mean()
        )
        group_df["GGF_std"] = (
            group_df["GGF_std"].rolling(ma_length, min_periods=1).mean()
        )

        plt.plot(group_df["step"], group_df["GGF_mean"], label=alg, alpha=0.8)
        plt.fill_between(
            group_df["step"],
            group_df["GGF_mean"] - group_df["GGF_std"],
            group_df["GGF_mean"] + group_df["GGF_std"],
            alpha=0.2,
        )

    plt.plot(
        group_df["step"],
        62 * np.ones(len(group_df["step"])),
        "r",
        label="Optimal",
        alpha=0.6,
    )
    plt.xlabel("Number of Steps")
    plt.ylabel("GGF Score")
    plt.legend(ncol=4, loc="lower right")
    plt.show()


pattern_list = ["sb3_reward_dqn", "sb3_reward_dqn_ggi"]
keys_list = ["DQN", "GGF-DQN"]
result_df = read_result_data(pattern_list, keys_list)
result_df_filter = result_df[result_df["step"] % 10 == 0]
plot_ggf(result_df_filter, keys_list)
