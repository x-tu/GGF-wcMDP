from src.solver.ggf_dual import build_ggf, extract_results, solve_ggf
from src.utils.mrp_lp import MRPData

# Get MRP data
input_data = MRPData(n_group=2, n_state=3, n_action=2, weight=[0.75, 0.25])
# Build the GGF model
ggf_model = build_ggf(data=input_data)
# Solve the GGF model
results, ggf_model = solve_ggf(model=ggf_model)
# Extract the results
extract_results(model=ggf_model, data=input_data)
