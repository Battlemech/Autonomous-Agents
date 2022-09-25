from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

from franka_move_task import FrankaMoveTask

# make sure only one franka is used for this sumulation script
task = FrankaMoveTask(name="Franka")
task.num_envs = 1

env.set_task(task, backend="torch")

from stable_baselines3 import PPO

import torch

# load local model
path = "ppo_franka"
model = PPO.load(path, env=env)
model.set_parameters(path)
env._world.reset()

# get target cube
cube = env._world.scene.get_object('target_cube0')

while env._simulation_app.is_running():

    # get current goal
    obs = task.get_observations()
    action, _states = model.predict(obs)
    task._frankas.set_joint_position_targets(torch.tensor(action))
    
    for _ in range(200):
        env._world.step(render=True)

    print("Resetting target")
    task.reset()

env.close()