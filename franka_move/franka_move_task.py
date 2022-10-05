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
from omni.isaac.core.utils.types import ArticulationActions

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
        self._max_speed = 5.0
        self._goal_tolerance = 0.2
        self._max_target_distance = 1.0
        self._ground_offset = 0

        # values used for defining RL buffers
        self._num_observations = 7 # 3 * goal coordinates + 4 * goal rotation
        self._num_actions = 9 # 9 rotor actuations
        self._device = "cpu"
        self.num_envs = 1

        # buffers to store RL data
        self.resets = torch.zeros((self.num_envs, 1))  # numer of resets

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
            position = np.array([0, index * self._max_target_distance * 2, self._ground_offset])

            # add cube with disabled collision
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
            target, rotation = self.generate_random_target_state()
            # set goal in simulation space
            self._target_cubes.set_world_poses(target, rotation, indices=[index]) #todo: more efficient?

        #self.targets = (torch.rand((num_resets, 3)) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance
    
        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # transform actions into force vectors
        actions = torch.tensor(actions).reshape(1, -1)
        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # apply them to the robots
        self._frankas.set_joint_positions(actions, indices=indices)
    
    def get_observations(self):
        # dof_vel = self._frankas.get_joint_velocities()
        # dof_finger_pos = self._franka_fingers.get_local_poses()[0] # get positions, ignore rotations
        # targets = self._target_cubes.get_local_poses()

        return torch.concat(self._target_cubes.get_local_poses(), dim=1)

    def calculate_metrics(self) -> None:
        # calculate distances to target cube
        distances_space, distances_rotation = self.calculate_distances()

        # check if robot reached goal
        resets_goal = torch.where(distances_space <= self._goal_tolerance, 1, 0)
        targets_reached = torch.sum(resets_goal)

        # track success and failure
        self.target_reached_count += targets_reached
        self.failure_count += self.num_envs - targets_reached

        # reward (left) being close to goal
        distance_metric = -4 * distances_space.double()
        rotation_metric = -distances_rotation.double()

        # return a malus if a invalid configuration was found
        return torch.where(torch.isnan(distances_space), torch.tensor(-self._max_target_distance ** 2, dtype=torch.double), distance_metric + rotation_metric).item()

    def is_done(self) -> None:
        # reset franka after one iteration
        return torch.ones(self.num_envs).item()

    def calculate_distances(self):
        # get positions and rotations of objects
        cube_pos, cube_rot = self._target_cubes.get_local_poses()
        franka_l_pos, franka_l_rotation = self._franka_fingers_left.get_local_poses()
        franka_r_pos, _ = self._franka_fingers_right.get_local_poses()

        # calculate point between left and right finger
        finger_middle_pos = (franka_l_pos + franka_r_pos) / 2

        # calculate distance (space and rotation)
        distance_space = torch.norm(finger_middle_pos - cube_pos, dim=1)
        # ISAAC GYM uses quaternions as orientation, calculate the angles between cubes and hands
        distance_rotation = torch.arccos(2 * (cube_rot @ franka_l_rotation.T).diagonal() ** 2 - 1)
        return distance_space, distance_rotation

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

