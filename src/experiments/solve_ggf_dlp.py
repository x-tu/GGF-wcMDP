from experiments.configs.base import params
from solver.ggf_dlp import build_dlp, extract_dlp, solve_dlp
from utils.common import MDP4LP
from utils.mrp import MRPData

mrp_data = MRPData(
    num_groups=params.num_groups,
    num_states=params.num_states,
    num_actions=params.num_actions,
    prob_remain=params.prob_remain,
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
)
model = build_dlp(mdp=mdp)
results, model = solve_dlp(model)
extract_dlp(model, mdp)
