# ISAAC core imports
from asyncio.log import logger
from cgitb import reset
from cmath import isnan

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
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

class FrankaMoveTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        # task-specific parameters
        self._goal_tolerance = 0.05
        self._max_target_distance = 1.0

        # task precision parameters
        self._max_step_count = 300 # maximum amount of steps per franka simulation
        self._joint_movement_tolerance = 0.001 # maximum amount of franka position which may change during a simulation until it is cocidered done

        # values used for defining RL buffers
        self._num_observations = 7 # 3 * goal coordinates + 4 * goal rotation values
        self._num_actions = 9 # 9 rotor actuations
        self._device = "cuda:0"
        #self.num_envs = 1
        self._num_envs = self._task_cfg["env"]["numEnvs"]

        # buffers to store RL data
        self.obs = torch.zeros((self.num_envs, self._num_observations))  # observations
        self.last_observation = torch.zeros((self.num_envs, self._num_observations))  # observations from previous step, used for exception handling
        self.reset_buf = torch.zeros((self.num_envs, 1))  # numer of resets
        self.actions = torch.zeros((self.num_envs, self._num_actions)) # actions of current simulation step

        # action and observation space
        # self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.action_space = spaces.Box(JOINT_LIMITS[:,0], JOINT_LIMITS[:,1])

        # todo: define more precise observation space
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # init parent class
        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"

        # create one target cube for each franka
        for index in range(self.num_envs):
            # add target cubes with disabled collision
            position = np.array([0, index * self._max_target_distance * 2, 0])
            
            # add the Franka USD to our stage
            create_prim(prim_path="/World/Franka" + str(index), prim_type="Xform", position=position)
            add_reference_to_stage(usd_path, "/World/Franka" + str(index))

            # add target cube
            cube = FixedCuboid(
                prim_path="/World/Franka" + str(index) + "/target_cube" + str(index),
                name="target_cube" + str(index),
                position=position,
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([0, 0, 1.0]),
            )
            scene.add(cube)
            cube.set_collision_enabled(False)
            # scene.add(Franka(
            #    prim_path="/World/Franka",
            #    name="franka" + str(index),
            #    position=np.array([0, index * self._max_target_distance * 2, 0])
            #))

        super().set_up_scene(scene)

        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._frankas = ArticulationView(prim_paths_expr="/World/Franka*", name="frankas_view")
        self._franka_fingers = ArticulationView(prim_paths_expr="/World/Franka*/panda_rightfinger", name="franka_fingers_view")
        self._target_cubes = ArticulationView(prim_paths_expr="/World/Franka*/target_cube*", name="target_view")

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
            target = ((torch.rand(1, 3) - torch.tensor([0.5, 0.5, 0])) * self._max_target_distance) + torch.tensor([0, index * self._max_target_distance * 2, 0.1])
            orientation = torch.rand(1, 4) * torch.tensor([360, 360, 360, 0])

            self._target_cubes.set_world_poses(target, orientation, indices=[index]) #todo: more efficient?
    
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # transform actions into force vectors
        self.actions = actions.reshape(-1, 9)
        indices = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # unprecise, but much quicker simulation
        self._frankas.set_joint_positions(self.actions, indices=indices)
    
    def get_observations(self):
        cube_positions, cube_rotations = self._target_cubes.get_local_poses()

        self.obs_buf = torch.concat((cube_positions, cube_rotations), 1)

        # return observations according to framework
        observations = {
            self._frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def calculate_metrics(self) -> None:
        # get poses
        franka_poses = self._franka_fingers.get_local_poses()
        target_poses = self._target_cubes.get_local_poses()

        distances_space = torch.norm(franka_poses[0] - target_poses[0], dim=1)
        distances_orientation = torch.norm(franka_poses[1] - target_poses[1], dim=1)

        # check if robot reached goal
        # resets_goal = torch.where(distances_space <= self._goal_tolerance, 1, 0)

        # reward (left finger) being close to goal
        distance_space_metric = (-4 * distances_space).double()
        distance_orientation_metric = -distances_orientation.double()

        # return malus if invalid configutation was found
        invalid_configurations = torch.isnan(self._frankas.get_joint_positions()).any(axis=1) # keepDims?

        self.rew_buf = torch.where(invalid_configurations == True, -self._max_target_distance ** 2, distance_space_metric + distance_orientation_metric)

    def is_done(self) -> None:
        # reset franka after one iteration
        self.reset_buf = torch.ones(self.num_envs)
