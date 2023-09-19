# Fair-RL

## Code Structure

```buildoutcfg
├── src                         # source code for the project
│   ├── env                     # RL environments
│   │   ├── predator_prey       # [To be Maintained] Predator-Prey environment (was rewritten based on ICML 2020 paper)
│   │   ├── mrp_env.py          # Machine Replacement Problem environment
│   │   └── mrp_env_rccc.py     # Machine Replacement Problem environment with Replacement Cost Constant Coefficient
│   ├── experiments             # main scripts for running experiments
│   │   ├── analysis            # functions used for analyzing/visualizing the results
│   │   ├── configs             # configurations for creating the RL agents
│   │   ├── results             # saved intermediate results of the experiments
│   │   ├── tests               # unit tests for the project
│   │   └── .py/.ipynb          # main scripts for running experiments
│   ├── solver                  # [In Progress] solver for solving the LP model
│   │   ├── fix_policy          # [To be added] solve the GGF-MDP(D) model with a fixed policy from RL
│   │   ├── ggf_dual            # solve the GGF-MDP(D) model
│   │   └── momdp               # solve the MOMDP model (no fairness)
│   ├── stable_baselines        # RL algorithms (stable-baselines, TensorFlow 1.x version)
│   ├── stable_baselines3       # RL algorithms (stable-baselines3, PyTorch version)
│   └── utils                   # shared useful functions/classes
└── requirements.txt            # all the packages needed for the project
```

**Official document for Python library used**:
1. [stable-baselines](https://stable-baselines.readthedocs.io/en/master/index.html)
2. [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
3. [pyomo](http://www.pyomo.org/)


## Package Setup

### Virtual Environment

`virtualenv` is a tool to create isolated Python environments, which reduces the dependencies
incompatible issue and makes it easier to manage multiple projects at the same time.

**Installation**

To install `virtualenv` via `apt`,

```
sudo apt install virtualenv
```

To install `virtualenv` via `pip`,

```
pip install virtualenv

# Or pip3 if you are using python 3
pip3 install virtualenv
```

### Quick Setup
Copy the commands below and run them.

```
# Setup a virtual environment for the project
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install all the packages via pip
pip install -r requirements.txt
# Or pip3 if you are using python 3
pip3 install -r requirements.txt
```

## Run code on the branch

To run code other than the main branch, for example, on a remote branch called `remote-branch-name`:
```
# Get the latest update
git pull
# Switch to the remote branch
git checkout remote-branch-name
```
