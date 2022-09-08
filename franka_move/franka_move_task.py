# ISAAC core imports
from cgitb import reset
from cmath import isnan
import re
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.objects import FixedCuboid

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
        self._max_speed = 3.0
        self._goal_tolerance = 0.1
        self._max_target_distance = 1.0
        self._reset_after = 100

        # values used for defining RL buffers
        self._num_observations = 20 # 7 rotor states [0, 2*Pi] + 7 * rotor accelerations + 3 * current coordinates of finger + 3 * goal coordinates
        self._num_actions = 7 # 7 rotor actuations
        self._device = "cpu"
        self.num_envs = 1

        # buffers to store RL data
        self.obs = torch.zeros((self.num_envs, self._num_observations))  # observations
        self.last_observation = torch.zeros((self.num_envs, self._num_observations))  # observations from previous step, used for exception handling
        self.resets = torch.zeros((self.num_envs, 1))  # numer of resets
        self.targets = torch.zeros((self.num_envs, 3))  # targets relative to each franka
        self.timestep_count = torch.zeros((self.num_envs, 1)) # simulated timesteps since last reset

        # action and observation space
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
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

        # create one target cube for each franka
        for _ in range(self.num_envs):
            scene.add(FixedCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=np.array([0, 0, -1.0]),
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([0, 0, 1.0]),
            ))

        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._frankas = ArticulationView(prim_paths_expr="/World/Franka*", name="frankas_view")
        self._franka_fingers = ArticulationView(prim_paths_expr="/World/Franka*/panda_rightfinger", name="franka_fingers_view")
        self._target_cubes = ArticulationView(prim_paths_expr="/World/target_cube*", name="target_view")

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

        # set each franka joint to a random degree # todo -> learn to start from any start
        # dof_pos = torch.rand(*(num_resets, self._frankas.num_dof), device=self._device) * np.pi * 2
        dof_pos = torch.zeros((num_resets, self._frankas.num_dof), device=self._device)

        # we init them with 0 for now
        dof_vel = torch.zeros((num_resets, self._frankas.num_dof), device=self._device)

        # apply franka resets
        indices = env_ids.to(dtype=torch.int32)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # generate goals only for robots which have been reset
        for index in env_ids:
            # generate goal
            target = (torch.rand(1, 3) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance
            # set goal info
            self.targets[index] = target
            # set goal in simulation space
            self._target_cubes.set_world_poses(target, indices=[index]) #todo: more efficient?

        #self.targets = (torch.rand((num_resets, 3)) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance
    
        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # transform actions into force vectors
        actions = torch.tensor(actions)

        # todo: assign in a more efficient way 
        # last 2 actions are always 0 -> We don't control the fingers
        forces = torch.zeros((self._frankas.count, self._frankas.num_dof), dtype=torch.float32, device=self._device)
        for i in range(len(self._joint_indices)):
            forces[:, self._joint_indices[i]] = self._max_speed * actions[i]

        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
    
        # apply them to the robots
        self._frankas.set_joint_efforts(forces, indices=indices)

        # increment amount of physics steps
        self.timestep_count += torch.ones((self.num_envs, 1))
    
    def get_observations(self):
        dof_pos = self._frankas.get_joint_positions()
        dof_vel = self._frankas.get_joint_velocities()
        dof_finger_pos = self._franka_fingers.get_local_poses()[0] # get positions, ignore rotations

        self.obs = torch.concat((dof_pos[:,:-2], dof_vel[:,:-2], dof_finger_pos, self.targets), dim=1)

        # check for NaN physics errors, reset robots
        self.resets = torch.where(torch.isnan(self.obs).any(axis=1, keepdims=True), 1, 0)
        # return observation from last iteration for frankas with all PhysX errors
        self.obs = torch.where(self.resets == 1, self.last_observation, self.obs)

        # update last observation
        self.last_observation = self.obs

        # return pos, velocity, finger position, goal
        return self.obs

    def calculate_metrics(self) -> None:
        dist = -self.calculate_distances()
        return dist.item()

    def is_done(self) -> None:
        # reset the robot if finger is in target region
        resets = torch.where(self.calculate_distances() <= self._goal_tolerance, 1, self.resets)
        # reset the robot if too many timespeps have passed in attempt to reach goal
        resets = torch.where(self.timestep_count >= self._reset_after, 1, resets)

        # reset timestep count of reset robots to 0
        self.timestep_count = (torch.ones((self.num_envs, 1)) - (resets)) * self.timestep_count

        self.resets = resets

        return resets.item()

    def calculate_distances(self):
        dof_finger_pos = self.obs[:, 14:17]
        dof_targets = self.obs[:, 17:]

        return torch.norm(dof_finger_pos - dof_targets, dim=1)
