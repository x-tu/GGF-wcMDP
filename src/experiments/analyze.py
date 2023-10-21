"""This script is used to analyze computational time."""

import json

import pandas as pd

from utils.common import DotDict

file_name = "time_stat_dqnS_4.json"
# notice: modify this folder name to the one containing the json file
folder = "results/1020_time_analysis/"
with open(f"{folder}{file_name}") as f:
    data = json.load(f)
data = DotDict(data)

index = [
    "total",
    "op_episode",
    "op_sample",
    "op_time_step",
    "op_ts_inner",
    "act",
    "> check_deterministic",
    "> solve_LP",
    "env",
    "improve",
    "> check_deterministic",
    "> solve_LP",
]
sec = [
    data.total,
    data.total - sum(data.episode),
    sum(data.episode) - sum(data.sample),
    sum(data.sample) - sum(data.step),
    -sum(data.step) + (sum(data.act) + sum(data.env) + sum(data.improve)),
    sum(data.act),
    sum(data.check_dtm_act),
    sum(data.solve_lp_act),
    sum(data.env),
    sum(data.improve),
    sum(data.check_dtm_improve),
    sum(data.solve_lp_improve),
]
percentage = [round(t / data.total, 4) * 100 for t in sec]

stat_df = pd.DataFrame({"%": percentage, "s": sec}, index=index)
print(stat_df)
