# Fair-RL

## Code Structure

```buildoutcfg
├── algorithms              # Baseline and other algorithms
│   ├── dlp                 # MC simulation with optimal solutions from GGF-LP model
│   └── whittle             # Whittle Index baseline
├── env                     # RL environments
│   ├── mrp_env_rccc        # Machine Replacement Problem (MRP) environment
│   └── mrp_simulation      # MRP env with count MDP
├── experiments             # main scripts for running experiments
│   ├── configs             # configurations for creating the RL agents
│   ├── batch_run           # train RL agents
│   ├── plot_figures        # plot figures
│   ├── run_whittle         # run Whittle Index baseline
│   └── solve_ggf_dlp       # solve the GGF-LP model
├── solver                  # LP solvers
│   ├── count_dlp           # solve the Count dual LP model
│   ├── dual_q              # solve GGF values based on Q values from RL agents
│   └── ggf_dual            # solve the GGF-MDP(D) model
├── stable_baselines3       # RL algorithms (stable-baselines3, PyTorch version)
├── utils                   # shared useful functions/classes
└── requirements.txt        # all the packages needed for the project
```

**Official document for Python library used**:
1. [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
2. [pyomo](http://www.pyomo.org/)


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
