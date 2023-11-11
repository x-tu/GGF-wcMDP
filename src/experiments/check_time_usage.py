"""This script is used to analyze the time usage of the LP model.

Specifically, we want to know the time usage of the following parts:
1. Run K times, each time build the model and solve the model.
   We then analyze the average time usage of (a) build the model (b) solve the model and (c) extract model results.
2. Build the Model only once, and update parameters K-1 times, analyze how much time we can save.
3. Remove one constraint, and analyze how much time we can save.

"""

from datetime import datetime

import numpy as np
from pandas import DataFrame as df
from pyomo.opt import SolverFactory
from tqdm import tqdm

from solver.dual_q import build_dual_q_model
from utils.common import DotDict

# set up the parameters
K_SIM = 10000
NUM_GROUPS = 4
# set the weights to be uniform
WEIGHTS = [1 / NUM_GROUPS] * NUM_GROUPS

# set up the time statistics
time_stat = DotDict(
    {
        "build": [],
        "update_param": [],
        "solve": [],
        "solve_update": [],
        "solve_partial": [],
        "extract": [],
    }
)

# Test 1: build and solve the model K times
for k in tqdm(range(K_SIM)):
    q_values = np.random.rand(NUM_GROUPS, NUM_GROUPS + 1)
    start_time = datetime.now()
    model = build_dual_q_model(q_values=q_values.tolist(), weights=WEIGHTS)
    # record the time usage for building the model
    time_stat.build.append((datetime.now() - start_time).total_seconds())
    # solve the model
    optimizer = SolverFactory("gurobi", solver_io="python")
    start_time = datetime.now()
    optimizer.solve(model, tee=False)
    # record the time usage for solving the model
    time_stat.solve.append((datetime.now() - start_time).total_seconds())

# Test 2: build the model once, and update the parameters K-1 times
q_values = np.random.rand(NUM_GROUPS, NUM_GROUPS + 1)
model = build_dual_q_model(q_values=q_values.tolist(), weights=WEIGHTS)
# solve the model
optimizer = SolverFactory("gurobi", solver_io="python")
for k in tqdm(range(K_SIM)):
    q_values = np.random.rand(NUM_GROUPS, NUM_GROUPS + 1)
    start_time = datetime.now()
    # Notice here we update the protected parameters
    model.qvalues._data.update(
        {
            (d, a): q_values[d][a]
            for d in range(NUM_GROUPS)
            for a in range(NUM_GROUPS + 1)
        }
    )
    # record the time for updating the parameters
    time_stat.update_param.append((datetime.now() - start_time).total_seconds())
    start_time = datetime.now()
    optimizer.solve(model, tee=False)
    # record the time usage after updating the parameters
    time_stat.solve_update.append((datetime.now() - start_time).total_seconds())
    start_time = datetime.now()
    policy = [model.varP[a].value for a in model.varP]
    # record the time usage after extracting the policy
    time_stat.extract.append((datetime.now() - start_time).total_seconds())

# Test 3: remove one constraint
q_values = np.random.rand(NUM_GROUPS, NUM_GROUPS + 1)
model = build_dual_q_model(q_values=q_values.tolist(), weights=WEIGHTS)
# remove the last constraint model.dual_constraints.add(sum(model.varP[a] for a in idx_list_a) == 1)
model.eq_constraints.deactivate()
# solve the model
optimizer = SolverFactory("gurobi", solver_io="python")
for k in tqdm(range(K_SIM)):
    q_values = np.random.rand(NUM_GROUPS, NUM_GROUPS + 1)
    model.qvalues.construct(q_values.tolist())
    start_time = datetime.now()
    optimizer.solve(model, tee=False)
    # record the time usage for solving the partial model
    time_stat.solve_partial.append((datetime.now() - start_time).total_seconds())

# save the time statistics
df.from_dict(time_stat, orient="index").to_csv(f"results/time_stat_{NUM_GROUPS}.csv")

print(sum(time_stat.build))
print(sum(time_stat.update_param))
print(sum(time_stat.solve))
print(sum(time_stat.solve_update))
print(sum(time_stat.solve_partial))
print(sum(time_stat.extract))
