"""Example instances for solving MOMDP model."""

from solver.momdp import solve_mrp
from utils.mrp import MRPData

# Get the MRP data
mrp_data = MRPData(n_group=3, n_state=3, n_action=2, weight=[1, 0.5, 0.25])
# Solve the MOMDP model
results, momdp_model = solve_mrp(input_data=mrp_data)
