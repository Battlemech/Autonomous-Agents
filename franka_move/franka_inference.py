from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

from franka_move_task import FrankaMoveTask

# make sure only one franka is used for this sumulation script
task = FrankaMoveTask(name="Franka")
task.num_envs = 1
# task._show_sample_space = True

env.set_task(task, backend="torch")

from stable_baselines3 import PPO

import torch

# load local model
path = "ppo_franka"
model = PPO.load(path, env=env)
model.set_parameters(path)
env._world.reset()

while env._simulation_app.is_running():
    for _ in range(300):
        # get current observations
        obs = task.get_observations()

        # get actions from model
        action, _states = model.predict(obs)
        action = torch.tensor(action)

        print("Goal configuration difference:", torch.norm(action - task._frankas.get_joint_positions()), "Goal distance:", task.calculate_distances())

        # apply actions to simulation
        task._frankas.set_joint_position_targets(action)
        env._world.step(render=True)

    print("Resetting target")
    task.reset()

env.close()
