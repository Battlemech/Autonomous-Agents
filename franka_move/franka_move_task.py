# ISAAC core imports
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim

# customise camer angle and viewport
import omni.kit

# reinforcement learning
import gym
from gym import spaces
import numpy as np
import torch
import math

class FrankaMoveTask(BaseTask):
    def __init__(self, name: str, offset= None) -> None:
        
        # task-specific parameters
        self._franka_position = [0.0, 0.0, 0.05]
        self._max_speed = 400.0
        self._goal_tolerance = 0.1
        self._max_target_distance = 0.8

        # values used for defining RL buffers
        self._num_observations = 20 # 7 rotor states [0, 2*Pi] + 7 * rotor accelerations + 3 * current coordinates of finger + 3 * goal coordinates
        self._num_actions = 7 # 7 rotor actuations
        self._device = "cpu"
        self.num_envs = 1

        # buffers to store RL data
        self.obs = torch.zeros((self.num_envs, self._num_observations))  # observations
        self.resets = torch.zeros((self.num_envs, 1))  # numer of resets
        self.targets = torch.zeros((self.num_envs, 3))  # targets relative to each franka

        # action and observation space
        self.action_space = spaces.Box(np.ones(self._num_actions) * 0.0, np.ones(self._num_actions) * np.pi * 2.0)
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # init parent class
        super().__init__(name, offset)

    def set_up_scene(self, scene) -> None:

        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        # add the Cartpole USD to our stage
        create_prim(prim_path="/World/Franka", prim_type="Xform", position=self._franka_position)
        add_reference_to_stage(usd_path, "/World/Franka")
        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._frankas = ArticulationView(prim_paths_expr="/World/Franka*", name="frankas_view")
        self._franka_fingers = ArticulationView(prim_paths_expr="/World/Franka*/panda_rightfinger", name="franka_fingers_view")

        # add Cartpole ArticulationView and ground plane to the Scene
        scene.add(self._frankas)
        scene.add_default_ground_plane()

        # save stage to allow looking up object positions #todo: better way?
        self._stage = scene.stage

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    def post_reset(self):
        # get joint indices for all joints
        self._joint_indices = [self._frankas.get_dof_index("panda_joint"+str(i)) for i in range(1, 8)]

        # randomize all envs
        indices = torch.arange(self._frankas.count, dtype=torch.int64, device=self._device)
        self.reset(indices)
    
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        # dof_pos = torch.zeros((num_resets, self._frankas.num_dof), device=self._device)
        # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # set each franka joint to a random degree
        dof_pos = torch.rand(*(num_resets, self._frankas.num_dof), device=self._device) * np.pi * 2

        # randomize DOF velocities
        # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # we init them with 0 for now
        dof_vel = torch.zeros((num_resets, self._frankas.num_dof), device=self._device)

        # generate goals
        self.targets = (torch.rand((self.num_envs, 3)) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # transform actions into force vectors
        actions = torch.tensor(actions)

        forces = torch.zeros((self._frankas.count, self._frankas.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._joint_indices[0]] = self._max_speed * actions[0]

        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
        # apply them to the robots
        self._frankas.set_joint_efforts(forces, indices=indices)
    
    def get_observations(self):
        dof_pos = self._frankas.get_joint_positions()
        dof_vel = self._frankas.get_joint_velocities()
        dof_finger_pos = self._franka_fingers.get_local_poses()[0] # get positions, ignore rotations

        self.obs = torch.concat((dof_pos[:,:-2], dof_vel[:,:-2], dof_finger_pos, self.targets), dim=1)

        # return pos, velocity, finger position, goal
        return self.obs

    def calculate_metrics(self) -> None:
        # dof_pos = self.obs[:, :7]
        # dof_vel = self.obs[:, 7:14]
        dof_finger_pos = self.obs[:, 14:17]
        dof_targets = self.obs[:, 17:]

        distance = torch.norm(dof_finger_pos - dof_targets, dim=1)

        print("Distance:", distance)

        return distance

    def is_done(self) -> None:
        # reset the robot if finger is in target region
        resets = torch.where(self.calculate_metrics() <= self._goal_tolerance, 1, 0)
        self.resets = resets

        # todo: reset if too many timesteps passed and goal was not reached

        return resets.item()