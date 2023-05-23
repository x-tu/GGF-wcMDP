import numpy as np
import pyomo.environ as pyo
from ggf_mdp import build_ggf, extract_results as extract_results_ggf
from mrp_data import MRPData

r3, r4 = [], []
# TODO: get rid of using parameters explicitly
policy = np.zeros((9, 3))
for i in range(100):
    # Get data
    input_data = MRPData(n_group=2, n_state=3, n_action=2, weight=[0.75, 0.25])
    ggf_model = build_ggf(data=input_data)
    pyo.SolverFactory("cbc").solve(ggf_model).write()
    reward_ggf, policy_ggf = extract_results_ggf(model=ggf_model, data=input_data)
    policy = policy + policy_ggf
    r3.append(reward_ggf[0])
    r4.append(reward_ggf[1])

policy_n = policy / 100
print(policy)
