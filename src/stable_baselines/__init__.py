import os

from stable_baselines.a2c import A2C
from stable_baselines.a2c_ggi import A2C_GGI
from stable_baselines.common import set_global_seeds
from stable_baselines.deepq import DQN
from stable_baselines.deepq_ggi import DQN_GGI
from stable_baselines.ppo2 import PPO2
from stable_baselines.ppo2_ggi import PPO2_GGI

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None
__version__ = "2.9.0a0"
