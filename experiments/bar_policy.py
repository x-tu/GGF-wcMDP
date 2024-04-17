import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, solve_dlp
from utils.common import MDP4LP
from utils.encoding import state_int_index_to_vector, state_vector_to_int_index
from utils.mrp import MRPData
from utils.permutation import generate_permutation_matrix_group
from utils.policy import (
    calculate_state_value,
    simulate_permuted_state_value,
    simulation_state_value,
)

params.num_groups = 3

mrp_data = MRPData(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    prob_remain=params.prob_remain,
    weight_type=params.weight_type,
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

state = np.array([0, 1, 2])
all_permutation_matrix = generate_permutation_matrix_group(params.num_groups)
initial_state_prob = np.zeros(mrp_data.num_global_states)
perm_state_list = [np.matmul(state, p_matrix) for p_matrix in all_permutation_matrix]
perm_state_idx_list = [
    state_vector_to_int_index(perm_state, params.num_states)
    for perm_state in perm_state_list
]
# calculate initial state probability
uniform_perm_state = [0] * mrp_data.num_global_states
for idx in perm_state_idx_list:
    uniform_perm_state[idx] += 1 / len(perm_state_idx_list)

# calculate LP values
model = build_dlp(mdp=mdp, initial_mu=uniform_perm_state)
# Solve the GGF model
_, model, _ = solve_dlp(model=model)
results = extract_dlp(model=model, print_results=True)
# get the policy
policy = results.policy

# initial state
params.num_samples = 5000
params.len_episode = 300
# simulate the permuted state value
Vt = simulate_permuted_state_value(
    params=params,
    policy=policy,
    mrp_data=mrp_data,
    initial_state_prob=uniform_perm_state,
    all_permutation_matrix=all_permutation_matrix,
)
