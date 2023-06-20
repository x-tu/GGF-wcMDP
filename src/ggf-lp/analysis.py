import pyomo.environ as pyo
import seaborn as sns
from ggf_dual import build_ggf, extract_results as extract_results_ggf
from matplotlib import pyplot as plt
from momdp import extract_results, solve_mrp
from mrp_data import MRPData


def compare_fairness(prob: float, n_experiment: int) -> None:
    """ This function is used to compare results

    Args:
        prob: the probability of transition staying at the current state.
        n_experiment: the number of experiments we run.

    Returns:
        None

    """
    r1, r2 = [], []
    r3, r4 = [], []
    for i in range(n_experiment):
        # Get data
        input_data = MRPData(n_group=2, n_state=3, n_action=2, weight=[prob, 1 - prob])
        # Results from the MOMDP model
        lp_model = solve_mrp(data=input_data)
        pyo.SolverFactory("cbc").solve(lp_model).write()
        reward = extract_results(model=lp_model, data=input_data)
        r1.append(reward[0])
        r2.append(reward[1])

        # Results from the GGF-MDP model
        ggf_model = build_ggf(data=input_data)
        pyo.SolverFactory("cbc").solve(ggf_model).write()
        reward_ggf, _ = extract_results_ggf(model=ggf_model, data=input_data)
        r3.append(reward_ggf[0])
        r4.append(reward_ggf[1])

    plt.scatter(r1, r2, marker=".", label=f"MOMDP ({prob})")
    plt.scatter(
        r3, r4, marker="o", facecolors="none", edgecolors="r", label=f"GGF-MDP ({prob})"
    )
    plt.legend(loc="lower left")
    plt.xlim(int(min(r1 + r2 + r3 + r4)) - 1, int(max(r1 + r2 + r3 + r4)) + 1)
    plt.ylim(int(min(r1 + r2 + r3 + r4)) - 1, int(max(r1 + r2 + r3 + r4)) + 1)
    plt.xlabel("Cost for Group 1")
    plt.ylabel("Cost for Group 2")
    plt.show()


def calculate_difference(prob: float, n_experiment: int) -> None:
    """ This function is used to compare results

    Args:
        prob: the probability of transition staying at the current state.
        n_experiment: the number of experiments we run.

    Returns:
        None

    """
    r1, r2 = [], []
    r3, r4 = [], []
    difference = []
    for i in range(n_experiment):
        # Get data
        input_data = MRPData(n_group=3, n_state=3, n_action=2, weight=[1, 0.5, 0.25])
        # Results from the MOMDP model
        lp_model = solve_mrp(data=input_data)
        pyo.SolverFactory("cbc").solve(lp_model).write()
        reward = extract_results(model=lp_model, data=input_data)
        r1.append(reward[0])
        r2.append(reward[1])

        # Results from the GGF-MDP model
        ggf_model = build_ggf(data=input_data)
        pyo.SolverFactory("cbc").solve(ggf_model).write()
        reward_ggf, _ = extract_results_ggf(model=ggf_model, data=input_data)
        r3.append(reward_ggf[0])
        r4.append(reward_ggf[1])
        difference.append(sum(reward) - sum(reward_ggf))

    sns.histplot(x=difference, kde=True)
    plt.xlabel("Price of Fairness")
    plt.show()


# compare_fairness(prob=0.75, n_experiment=100)
calculate_difference(prob=0.75, n_experiment=1000)
