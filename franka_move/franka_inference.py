from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True)

from franka_move_task import FrankaMoveTask
task = FrankaMoveTask(name="Franka")
env.set_task(task, backend="torch")

from stable_baselines3 import PPO

# run inference on the trained policy
path = "ppo_franka"
model = PPO.load(path)
env._world.reset()
obs = env.reset()