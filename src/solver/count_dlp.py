import random
from datetime import datetime

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from utils.count import CountMDP
from experiments.configs.base import params


def build_count_dlp(
        count_mdp: CountMDP,
        deterministic_policy: bool = False, initial_mu: list = None
) -> pyo.ConcreteModel:
    """Used to build the GGF dual MDP (stochastic) model."""

    model = pyo.ConcreteModel()
    model.mdp = count_mdp
    model.discount = params.gamma

    # Indices
    state_indices = range(count_mdp.num_count_states)
    action_indices = range(count_mdp.num_count_actions)
    model.init_distribution = [0.25, 0.5, 0.25]
    # Variables
    model.varX = pyo.Var(
        state_indices, action_indices, within=pyo.NonNegativeReals
    )

    mdp = None
    # Objective
    model.objective = pyo.Objective(
        expr=sum(
            sum(
                count_mdp.count_costs[s, a] * model.varX[s, a]
                for a in action_indices
            )
            for s in state_indices
            )
        ,
        sense=pyo.minimize,
    )

    # Constraints
    model.dual_constraints = pyo.ConstraintList()
    # Group 2 (s ^ D Constraints)
    for s in state_indices:
        model.dual_constraints.add(
            sum(model.varX[s, a1] for a1 in action_indices)
            - model.discount
            * (
                sum(
                    model.varX[j, a2] * count_mdp.count_transitions[j, s, a2]
                    for j in state_indices
                    for a2 in action_indices
                )
            )
            == model.init_distribution[s]
        )
    return model


def solve_count_dlp(model: pyo.ConcreteModel, num_opt_solutions: int = 1):
    """ Selects the solver and set the optimization settings.

    Args:
        model: the MRP model to be optimized
        num_opt_solutions: the number of optimal solutions to be returned

    Returns:
        results: the default optimization report
        model: the optimized model
        all_solutions: the list of all sub-solutions found

    """
    # Set the solver to be used
    opt = SolverFactory("gurobi_persistent", solver_io="python")
    opt.set_instance(model)

    if num_opt_solutions > 1:
        opt.set_gurobi_param("PoolSolutions", num_opt_solutions)
        opt.set_gurobi_param("PoolSearchMode", 2)

    # Solve the model
    results = opt.solve(model, tee=False, report_timing=False)
    return results, model


def extract_count_dlp(model: pyo.ConcreteModel, print_results: bool = False):
    """ This function is used to extract optimized results.

    Args:
        model: the MRP model to be optimized
        print_results: whether to print the results

    Returns:
        results: the extracted results

    """
    var_x_np = np.zeros((model.mdp.num_count_states, model.mdp.num_count_actions))
    for s in range(model.mdp.num_count_states):
        for a in range(model.mdp.num_count_actions):
            var_x_np[s, a] = model.varX[s, a].value
    objective = model.objective()
    return results


count_MDP = CountMDP(num_groups = 2, num_states = 2, num_actions = 2)
model = build_count_dlp(count_MDP, deterministic_policy = False, initial_mu = None)
results, model = solve_count_dlp(model, num_opt_solutions = 1)
results = extract_count_dlp(model, print_results = True)
print(results)