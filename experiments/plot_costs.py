import matplotlib.pyplot as plt

from experiments.configs.base import params
from utils.mrp import MRPData

costs = ["zero", "constant", "linear", "quadratic", "exponential", "rccc"]
x_range = range(params.num_states)
for op_cost_idx in range(len(costs)):
    for rp_cost_idx in range(len(costs)):
        # plot subfigures
        mrp = MRPData(
            num_groups=params.num_groups,
            num_states=params.num_states,
            cost_types_operation=costs[op_cost_idx],
            cost_types_replace=costs[rp_cost_idx],
        )
        plt.subplot(6, 6, op_cost_idx * 6 + rp_cost_idx + 1)

        # operation cost for each group
        operation_cost = mrp.costs[0, :, 0]
        # repair cost for each group
        repair_cost = mrp.costs[0, :, 1]

        plt.plot(x_range, operation_cost)
        plt.plot(x_range, repair_cost)

        # no x and y ticks
        plt.xticks([])
        plt.yticks([])

# label only the outer plots
for i in range(6):
    plt.subplot(6, 6, i * 6 + 1)
    plt.ylabel(f"{costs[i]}")

    plt.subplot(6, 6, 6 * 5 + 1 + i)
    plt.xlabel(f"{costs[i]}")
plt.savefig("experiments/tmp/operation_repair_cost.png")
plt.show()
