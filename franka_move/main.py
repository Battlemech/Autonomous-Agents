from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True)

from franka_move_task import FrankaMoveTask
task = FrankaMoveTask(name="Franka")
env.set_task(task, backend="torch")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from os.path import exists

timesteps = 100000
path = "ppo_franka"

# log success rate to tensor board
class TensorBoardCallback(BaseCallback):
    def __init__(self) -> None:
        super(TensorBoardCallback, self).__init__()

    def _on_step(self) -> bool:
        if task.target_reached_count == 0 or task.failure_count == 0:
                return True

        # reset target_reached count and failure count if sum gets to high -> Accuratly display new attempts
        if task.target_reached_count + task.failure_count >= 100:
                task.target_reached_count = task.target_reached_count / 25
                task.failure_count = task.failure_count / 25

        self.logger.record('Success rate', task.target_reached_count / (task.target_reached_count + task.failure_count))
        return True


# try loading old model. OnFail: create new one
if exists(path):
        model = PPO.load(path)
else:
        # create agent from stable baselines
        model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
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

while True:
        model.learn(total_timesteps=timesteps, callback=TensorBoardCallback())
        model.save("ppo_franka")

env.close()