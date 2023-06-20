# Fair-RL

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

## Code Structure

**LP-based methods**: at the folder `src/ggf-lp`.
+ `ggf_dual`: an implementation for GGF-MDP dual formulation
+ `momdp`: not really on vector optimization, but a weighted sum of objectives

**RL environment**
+ machine replacement environment at the folder `src/machine-ggf`
+ predator-prey environment at the folder `src/predator_prey` (from the ICML 2020 paper)

**RL algorithms**: at the folder `src/stable_baseline`
+ contains the implementation of the RL algorithms adapted based on the Python library [stable-baseline](https://stable-baselines.readthedocs.io/en/master/index.html)
