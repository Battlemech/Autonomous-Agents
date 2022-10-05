# ISAAC core imports
from asyncio.log import logger
from cgitb import reset
from cmath import isnan

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.franka import Franka
from omni.isaac.core import World

# customise camer angle and viewport
import omni.kit

# reinforcement learning
import gym
from gym import spaces
import numpy as np
import torch

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
        self._goal_tolerance = 0.2 # range: [0, 1]
        self._franksas_offset = 2.0 # spacing between multiple frankas
        self._show_sample_space = False
        self._reset_after = 200 # After after how many tries a new target is generated
        
        # values used for defining RL buffers
        self._num_observations = 7 # 3 * goal coordinates + 4 * goal rotation values
        self._num_actions = 9 # 9 rotor actuations
        self._device = "cpu"
        self.num_envs = 1

        # buffers to store RL data
        self.obs = torch.zeros((self.num_envs, self._num_observations))  # observations
        self.timestep_count = torch.zeros((self.num_envs, 1))
        self.resets = torch.zeros((self.num_envs, 1))  # numer of resets
        self.actions = torch.zeros((self.num_envs, self._num_actions)) # actions of current simulation step

        # buffers to store debug data
        self.target_reached_count = 0
        self.failure_count = 0

        # action and observation space
        # self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.action_space = spaces.Box(JOINT_LIMITS[:,0], JOINT_LIMITS[:,1])

        # todo: define more precise observation space
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # init parent class
        super().__init__(name, offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"

        # create one target cube for each franka
        for index in range(self.num_envs):
            # add target cubes with disabled collision
            position = np.array([0, index * self._franksas_offset * 2, 0])
            cube = FixedCuboid(
                prim_path="/World/target_cube" + str(index),
                name="target_cube" + str(index),
                position=position,
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([0, 0, 1.0]),
            )
            scene.add(cube)
            cube.set_collision_enabled(False)

            # add the Franka USD to our stage
            create_prim(prim_path="/World/Franka" + str(index), prim_type="Xform", position=position)
            add_reference_to_stage(usd_path, "/World/Franka" + str(index))

        # generate debug targets in sample space
        if self._show_sample_space:
            for i in range(1, 200):
                pos, ori = self.generate_random_target_state()
                cube = FixedCuboid(
                    prim_path="/World/dummy_cube" + str(i),
                    name="dummy_cube" + str(i),
                    position=pos,
                    scale=np.array([0.5, 0.5, 0.5]),
                    color=np.array([0, 1.0, 0]),
                )
                cube.set_collision_enabled(False)
                scene.add(cube)

        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._frankas = ArticulationView(prim_paths_expr="/World/Franka*", name="frankas_view")
        self._franka_fingers_right = ArticulationView(prim_paths_expr="/World/Franka*/panda_rightfinger", name="franka_fingers_view_right")
        self._franka_fingers_left = ArticulationView(prim_paths_expr="/World/Franka*/panda_leftfinger", name="franka_fingers_view_left")
        self._target_cubes = ArticulationView(prim_paths_expr="/World/target_cube*", name="target_view")

        # add Cartpole ArticulationView and ground plane to the Scene
        scene.add(self._frankas)
        scene.add_default_ground_plane()

        # save stage to allow looking up object positions #todo: better way?
        self._stage = scene.stage

        # set default camera viewport position and target
        self.set_initial_camera_params()

        # save current instance of world to allow performing steps
        self._world = World.instance()

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

        # apply franka resets
        indices = env_ids.to(dtype=torch.int32)
        # self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(torch.zeros((num_resets, self._frankas.num_dof), device=self._device), indices=indices)

        # generate goals only for robots which have been reset
        for index in env_ids:
            # generate goal
            target, orientation = self.generate_random_target_state()
            self._target_cubes.set_world_poses(target, orientation, indices=[index]) #todo: more efficient?

        #self.targets = (torch.rand((num_resets, 3)) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance
    
        # bookkeeping
        self.resets[env_ids] = 0

    def generate_random_target_state(self):
        # generate random direction (positive height)
        direction = torch.rand(1, 3) - torch.tensor([0.5, 0.5, 0.0])

        # norm it -> diection has length 1
        direction = (direction / torch.norm(direction))

        # set random length
        direction = direction * torch.rand(1) * 1.15

        # increase base height of target
        direction = direction + torch.tensor([0.0, 0.0, 0.1])

        orientation = torch.rand(1, 4) * torch.tensor([360, 360, 360, 0])
        return direction, orientation

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # transform actions into force vectors
        self.actions = torch.tensor(actions).reshape(1, -1)
        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # unprecise, but much quicker simulation
        self._frankas.set_joint_positions(self.actions, indices=indices)
        self.timestep_count += 1
    
    def get_observations(self):
        # get poses
        cube_pos, cube_ori = self._target_cubes.get_local_poses()
        franka_poses_right = self._franka_fingers_right.get_local_poses()
        franka_poses_left = self._franka_fingers_left.get_local_poses()

        # calculate coordinate between fingers
        between_fingers_positions = (franka_poses_right[0] - franka_poses_left[0]) / 2

        self.distances_space = torch.norm(between_fingers_positions - cube_pos, dim=1, keepdim=True)
        self.distances_orientation = torch.norm(franka_poses_right[1] - cube_ori, dim=1, keepdim=True)
        
        return torch.concat((cube_pos, cube_ori), dim=1)

    def calculate_metrics(self) -> None:
        # check if robot reached goal
        self.resets_goal = torch.where(self.distances_space <= self._goal_tolerance, 1, 0)
        targets_reached = torch.sum(self.resets_goal)

        # track success and failure
        self.target_reached_count += targets_reached
        self.failure_count += self.num_envs - targets_reached

        # reward (point between fingers) being close to goal
        distance_space_metric = (-4 * self.distances_space).double()
        distance_orientation_metric = -self.distances_orientation.double()

        # return malus if invalid configuration was found
        invalid_configurations = torch.isnan(self._frankas.get_joint_positions()).any(axis=1, keepdims=True)

        # malus for invalid configurations
        error_malus = torch.where(invalid_configurations == True, -4.0, distance_space_metric + distance_orientation_metric)

        # give higher reward for reaching goal
        return (error_malus + torch.where(self.resets_goal == True, 1, 0)).item()
        

    def is_done(self) -> None:
        # reset frankas which exceeded max timestep or reached goal
        return torch.where(self.timestep_count >= self._reset_after, 1, self.resets_goal)
