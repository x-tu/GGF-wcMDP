"""Example instances for solving GGF dual model."""
from src.solver.ggf_dual import solve_ggf
from src.utils.mrp import MRPData

# Get MRP data
mrp_data = MRPData(n_group=2, n_state=3, n_action=2, weight=[0.75, 0.25])
# Solve the GGF model
results, ggf_model = solve_ggf(input_data=mrp_data, solve_deterministic=False)
