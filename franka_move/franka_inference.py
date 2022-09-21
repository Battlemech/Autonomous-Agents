from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

from franka_move_task import FrankaMoveTask

# make sure only one franka is used for this sumulation script
task = FrankaMoveTask(name="Franka")
task.num_envs = 1

env.set_task(task, backend="torch")

from stable_baselines3 import PPO

import torch

# run inference on the trained policy
path = "ppo_franka"
model = PPO.load(path, env=env)
env._world.reset()

# get target cube
cube = env._world.scene.get_object('target_cube0')

while env._simulation_app.is_running():

    # get current goal
    obs = task.get_observations()
    action, _states = model.predict(obs)
    task._frankas.set_joint_position_targets(torch.tensor(action))
    env._world.step()

    distance = task.calculate_distances()[0]
    print("Distance:", distance, "Within goal tolerance:", distance <= task._goal_tolerance)

    # reset task if invalid state was reached
    if torch.any(torch.isnan(obs)):
        task.reset()
    #obs, rewards, dones, info = env.step(action)

env.close()