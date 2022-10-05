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

# log success rate to tensor board
class TensorBoardCallback(BaseCallback):
    def __init__(self) -> None:
        super(TensorBoardCallback, self).__init__()

    def _on_step(self) -> bool:
        # only start logging success rate after a few attempts have been made
        if task.target_reached_count + task.failure_count < 20:
                return True

        # reset target_reached count and failure count if sum gets to high -> Accuratly display new attempts
        if task.target_reached_count + task.failure_count >= 200:
                task.target_reached_count = task.target_reached_count / 2
                task.failure_count = task.failure_count / 2

        self.logger.record('Success rate', (task.target_reached_count / (task.target_reached_count + task.failure_count)).item())
        return True


# try loading old model. OnFail: create new one
if exists(path+".zip"):
        model = PPO.load(path)
        model.set_env(env)
        model.set_parameters(path)
        
        print("Loaded old model!", model)
else:
        # create agent from stable baselines
        model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=1024,
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

        print("Created new model!")

# save and close model on interrupt
def signal_handler(sig, frame):
        model.save(path)
        env.close()
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# learn for set amount of timesteps
for _ in range(10):
        model.learn(total_timesteps=timesteps/10, callback=TensorBoardCallback(), reset_num_timesteps=False)

# save model, close simulation
model.save(path)
env.close()