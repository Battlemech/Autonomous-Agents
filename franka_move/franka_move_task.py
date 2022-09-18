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
from omni.isaac.franka import Franka

# customise camer angle and viewport
import omni.kit

# reinforcement learning
import gym
from gym import spaces
import numpy as np
import torch
import math

JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
JOINT_LIMITS = np.array([[-2.8973,  2.8973],
         [-1.7628,  1.7628],
         [-2.8973,  2.8973],
         [-3.0718, -0.0698],
         [-2.8973,  2.8973],
         [-0.0175,  3.7525],
         [-2.8973,  2.8973],
         [ 0.0000,  0.0400],
         [ 0.0000,  0.0400]])

class FrankaMoveTask(BaseTask):
    def __init__(self, name: str, offset= None) -> None:
        # task-specific parameters
        self._max_speed = 5.0
        self._goal_tolerance = 0.2
        self._max_target_distance = 1.0
        self._reset_after = 200

        # values used for defining RL buffers
        self._num_observations = 24 # 9 rotor states [0, 2*Pi] + 9 rotor velocities + 3 * current coordinates of finger + 3 * goal coordinates
        self._num_actions = 9 # 9 rotor actuations
        self._device = "cpu"
        self.num_envs = 1

        # buffers to store RL data
        self.obs = torch.zeros((self.num_envs, self._num_observations))  # observations
        self.last_observation = torch.zeros((self.num_envs, self._num_observations))  # observations from previous step, used for exception handling
        self.resets = torch.zeros((self.num_envs, 1))  # numer of resets
        self.targets = torch.zeros((self.num_envs, 3))  # targets relative to each franka
        self.timestep_count = torch.zeros((self.num_envs, 1)) # simulated timesteps since last reset
        self.actions = torch.zeros((self.num_envs, self._num_actions)) # actions of current simulation step
        self.prev_actions = torch.zeros((self.num_envs, self._num_actions)) # actions of previous simulation step

        # bufferst to store dubug data
        self.target_reached_count = 0
        self.failure_count = 0

        # action and observation space
        # self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.action_space = spaces.Box(JOINT_LIMITS[:,0], JOINT_LIMITS[:,1])
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # init parent class
        super().__init__(name, offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"

        # create one target cube for each franka
        for index in range(self.num_envs):
            position = np.array([0, index * self._max_target_distance * 2, 0])
            scene.add(FixedCuboid(
                prim_path="/World/target_cube" + str(index),
                name="target_cube" + str(index),
                position=position,
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([0, 0, 1.0]),
            ))

            # add the Franka USD to our stage
            create_prim(prim_path="/World/Franka" + str(index), prim_type="Xform", position=position)
            add_reference_to_stage(usd_path, "/World/Franka" + str(index))

            # scene.add(Franka(
            #    prim_path="/World/Franka",
            #    name="franka" + str(index),
            #    position=np.array([0, index * self._max_target_distance * 2, 0])
            #))

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
        self._joint_indices = [self._frankas.get_dof_index(name) for name in JOINT_NAMES]

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

        # save action from previous iteration
        self.prev_actions = self.actions

        # transform actions into force vectors
        self.actions = torch.tensor(actions).reshape(1, -1)
        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # apply them to the robots
        self._frankas.set_joint_position_targets(self.actions, indices=indices)

        # increment amount of physics steps
        self.timestep_count += torch.ones((self.num_envs, 1))
    
    def get_observations(self):
        dof_pos = self._frankas.get_joint_positions()
        dof_vel = self._frankas.get_joint_velocities()

        # dof_vel = self._frankas.get_joint_velocities()
        dof_finger_pos = self._franka_fingers.get_local_poses()[0] # get positions, ignore rotations

        self.obs = torch.concat((dof_pos, dof_vel, dof_finger_pos, self.targets), dim=1)

        # check for NaN physics errors, reset robots
        self.resets = torch.where(torch.isnan(self.obs).any(axis=1, keepdims=True), 1, 0)
        # return observation from last iteration for frankas with all PhysX errors
        self.obs = torch.where(self.resets == 1, self.last_observation, self.obs)

        # update last observation
        self.last_observation = self.obs

        # return pos, velocity, finger position, goal
        return self.obs

    def calculate_metrics(self) -> None:
        distance_metric = (-(self.calculate_distances() ** 2)) * 2

        # give positive reward if goal was reached
        for i in range(len(distance_metric)):
            if(distance_metric[i] <= self._goal_tolerance):
                distance_metric[i] = (self._reset_after - self.timestep_count[i]) * 10

        return distance_metric.item()
        # action_metric = -(torch.sum(torch.abs(self.prev_actions - self.actions), dim=1) / (self._num_actions * 4))
        # return (distance_metric + action_metric).item()

    def is_done(self) -> None:
        # reset the robot if finger is in target region
        resets_goal = torch.where(self.calculate_distances() <= self._goal_tolerance, 1, 0)

        # reset the robot if too many timespeps have passed in attempt to reach goal
        resets_failure = torch.where(self.timestep_count >= self._reset_after, 1, 0)

        # get previous resets from physx errors
        resets = torch.where(resets_goal + resets_failure + self.resets >= 1, 1, 0)

        # reset timestep count of reset robots to 0
        self.timestep_count = (torch.ones((self.num_envs, 1)) - (resets)) * self.timestep_count

        self.resets = resets

        return resets.item()

    def calculate_distances(self):
        dof_finger_pos = self.obs[:, 18:21]
        dof_targets = self.obs[:, 21:]

        return torch.norm(dof_finger_pos - dof_targets, dim=1)
