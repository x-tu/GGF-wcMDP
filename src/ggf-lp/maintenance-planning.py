"""This is a toy example from Pyomo Cookbook, which is used as an example to test
    whether pyomo works as expected."""

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

# problem parameters
T = 90  # planning period from 1..T
M = 3  # length of maintenance period
P = 4  # number of maintenance periods

# daily profits
c = {k: np.random.uniform() for k in range(1, T + 1)}


def maintenance_planning_bigm(c, T, M, P):
    m = pyo.ConcreteModel()

    m.T = pyo.RangeSet(1, T)
    m.Y = pyo.RangeSet(1, T - M + 1)
    m.S = pyo.RangeSet(0, M - 1)

    m.c = pyo.Param(m.T, initialize=c)
    m.x = pyo.Var(m.T, domain=pyo.Binary)
    m.y = pyo.Var(m.T, domain=pyo.Binary)

    # objective
    m.profit = pyo.Objective(expr=sum(m.c[t] * m.x[t] for t in m.T), sense=pyo.maximize)

    # required number P of maintenance starts
    m.sumy = pyo.Constraint(expr=sum(m.y[t] for t in m.Y) == P)

    # no more than one maintenance start in the period of length M
    m.sprd = pyo.Constraint(m.Y, rule=lambda m, t: sum(m.y[t + s] for s in m.S) <= 1)

    # disjunctive constraints
    m.bigm = pyo.Constraint(m.Y, rule=lambda m, t: sum(m.x[t + s] for s in m.S) <= M * (1 - m.y[t]))

    return m


def plot_schedule(m):
    fig, ax = plt.subplots(3, 1, figsize=(9, 4))

    ax[0].bar(m.T, [m.c[t] for t in m.T])
    ax[0].set_title('daily profit $c_t$')

    ax[1].bar(m.T, [m.x[t]() for t in m.T], label='normal operation')
    ax[1].set_title('unit operating schedule $x_t$')

    ax[2].bar(m.Y, [m.y[t]() for t in m.Y])
    ax[2].set_title(str(P) + ' maintenance starts $y_t$')
    for a in ax:
        a.set_xlim(0.1, len(m.T) + 0.9)

    plt.tight_layout()
    plt.show()


m = maintenance_planning_bigm(c, T, M, P)
pyo.SolverFactory('cbc').solve(m).write()
plot_schedule(m)
