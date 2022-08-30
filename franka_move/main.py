from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

from franka_move_task import FrankaMoveTask
task = FrankaMoveTask(name="Franka")
env.set_task(task, backend="torch")

reset_count = 0

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        print("------- Reset")
        reset_count += 1
        observation = env.reset()
env.close()

print("Reset count:", reset_count, reset_count/1000, "%")

"""
from stable_baselines3 import PPO

# create agent from stable baselines
model = PPO(
        "MlpPolicy",
        env,
        n_steps=1000,
        batch_size=1000,
        n_epochs=20,
        learning_rate=0.001,
        gamma=0.99,
        device="cuda:0",
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        verbose=1,
        tensorboard_log="./franka_tensorboard"
)
model.learn(total_timesteps=1000)
model.save("ppo_franka")

env.close()
"""