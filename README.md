# Fair-RL

## Code Structure

```buildoutcfg
├── src                         # source code for the project
│   ├── envs                    # RL environments
│   │   ├── predator_prey       # [To be Maintained] Predator-Prey environment (rewritten based on the ICML 2020 paper)
│   │   └── mrp_env.py          # Machine Replacement Problem environment
│   ├── experiments             # main scripts for running experiments
│   │   ├── analysis            # functions used for analyzing/visualizing the results
│   │   ├── configs             # [In Progress] configurations for creating the RL agents
│   │   ├── main                # main scripts for running experiments
│   │   ├── results             # saved intermediate results of the experiments
│   │   └── exec_experiments.py # currently used as main script for running experiments
│   ├── solver                  # solver for solving the LP model
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
