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


