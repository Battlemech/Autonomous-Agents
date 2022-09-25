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
        self._goal_tolerance = 0.05
        self._max_target_distance = 1.0

        # task precision parameters
        self.precise_simulation = False # If false, joints will be instantly teleported to target and max_step_cpunt and joint_movement_tolerance will be ignored
        self._max_step_count = 300 # maximum amount of steps per franka simulation
        self._joint_movement_tolerance = 0.001 # maximum amount of franka position which may change during a simulation until it is cocidered done

        # values used for defining RL buffers
        self._num_observations = 7 # 3 * goal coordinates + 4 * goal rotation values
        self._num_actions = 9 # 9 rotor actuations
        self._device = "cpu"
        self.num_envs = 1

        # buffers to store RL data
        self.obs = torch.zeros((self.num_envs, self._num_observations))  # observations
        self.last_observation = torch.zeros((self.num_envs, self._num_observations))  # observations from previous step, used for exception handling
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
            position = np.array([0, index * self._max_target_distance * 2, 0])
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
            target = ((torch.rand(1, 3) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance) + torch.tensor([0, 0, 0.1])
            orientation = torch.rand(1, 4) * torch.tensor([360, 360, 360, 0])

            self._target_cubes.set_world_poses(target, orientation, indices=[index]) #todo: more efficient?

        #self.targets = (torch.rand((num_resets, 3)) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance
    
        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # transform actions into force vectors
        self.actions = torch.tensor(actions).reshape(1, -1)
        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # unprecise, but much quicker simulation
        if not self.precise_simulation:
            self._frankas.set_joint_positions(self.actions, indices=indices)
            return

        # apply them to the robots
        self._frankas.set_joint_positions(self.actions, indices=indices)
        self._frankas.set_joint_position_targets(self.actions, indices=indices)

        # simulate each franka joint, which allows observing if a configuration can be maintained or is invalid (due to physics/joint positions)
        old_positions = self._frankas.get_joint_positions()
        for _ in range(self._max_step_count):
            # simulate 1 step
            self._world.step(render=False)

            # get current joint positions in new step
            current_positions = self._frankas.get_joint_positions()
            
            # stop simulating this step if invalid configuration was reached
            if torch.any(torch.isnan(self._frankas.get_joint_positions())):
                return

            # terminate if simulated steps didn't change # todo: only continue simulating frankas whose joint position moved too much
            if torch.norm(current_positions - old_positions) <= self._joint_movement_tolerance:
                return

            # update current position
            old_positions = current_positions     
    
    def get_observations(self):
        poses = torch.tensor([])
        for pose in self._target_cubes.get_local_poses():
            poses = torch.cat((poses, pose[0]), 0)
        
        # get targets
        return poses

    def calculate_metrics(self) -> None:
        # get poses
        franka_poses = self._franka_fingers.get_local_poses()
        target_poses = self._target_cubes.get_local_poses()

        distances_space = torch.norm(franka_poses[0] - target_poses[0], dim=1)
        distances_orientation = torch.norm(franka_poses[1] - target_poses[1], dim=1)

        # check if robot reached goal
        resets_goal = torch.where(distances_space <= self._goal_tolerance, 1, 0)
        targets_reached = torch.sum(resets_goal)

        # track success and failure
        self.target_reached_count += targets_reached
        self.failure_count += self.num_envs - targets_reached

        # reward (left finger) being close to goal
        distance_space_metric = (-4 * distances_space).double()
        distance_orientation_metric = -distances_orientation.double()

        # return malus if invalid configutation was found
        invalid_configurations = torch.isnan(self._frankas.get_joint_positions()).any(axis=1, keepdims=True)

        return torch.where(invalid_configurations == True, -self._max_target_distance ** 2, distance_space_metric + distance_orientation_metric).item()

    def is_done(self) -> None:
        # reset franka after one iteration
        return torch.ones(self.num_envs).item()
