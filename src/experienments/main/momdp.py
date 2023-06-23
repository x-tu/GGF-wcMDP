from src.solver.momdp import build_mrp, extract_results, solve_mrp
from src.utils.mrp_lp import MRPData

# Get the MRP data
input_data = MRPData(n_group=3, n_state=3, n_action=2, weight=[1, 0.5, 0.25])
# Build the MOMDP model
momdp_model = build_mrp(data=input_data)
# Solve the MOMDP model
results, momdp_model = solve_mrp(model=momdp_model)
# Extract the results
extract_results(model=momdp_model, data=input_data)
