from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True)

from franka_move_task import FrankaMoveTask
task = FrankaMoveTask(name="Franka")
env.set_task(task, backend="torch")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from os.path import exists
import signal
import sys

timesteps = 1000000
path = "ppo_franka"