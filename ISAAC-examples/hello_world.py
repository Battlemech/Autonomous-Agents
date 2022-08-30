from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.tasks import BaseTask
import numpy as np

PRINT_AFTER_TICK_COUNT = 25
GOAL_TOLERANCE = 0.02
PRINT_PREFIX = "[AA]"

class FrankaPlaying(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

        # define goal
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        print(PRINT_PREFIX, "Goal position:", self._goal_position)

        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        # add ground plane
        scene.add_default_ground_plane()

        # add cube to be moved
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            size=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0, 0, 1.0])))

        # add default manipulator robot (franka)
        self._franka = scene.add(Franka(prim_path="/World/Fancy_Franka",
                                        name="fancy_franka"))
        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        # position of cube to be moved
        cube_position, _ = self._cube.get_world_pose()

        # joint positions of manipulator robot
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations

    # Called before each physics step,
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()

        # if task was not archived (yet) and cube is in the vicinity of goal
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < GOAL_TOLERANCE:
            # set rgb values of cube
            # todo: fix
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # track tick count for debug purposes
        self._tick_count = 0

        return

    def setup_scene(self):
        world = self.get_world()
        # Add move object task to world
        world.add_task(FrankaPlaying(name="my_first_task"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task (with first reset of the world)
        # so we can retrieve the task objects
        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper_dof_indices=self._franka.gripper.dof_indices,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()

        # print observations to console
        if self._tick_count % PRINT_AFTER_TICK_COUNT == 0:
            print(PRINT_PREFIX, "State:", current_observations)

        # increment tick count
        self._tick_count = self._tick_count + 1

        # forward observations to controler
        actions = self._controller.forward(
            picking_position=current_observations["fancy_cube"]["position"],
            placing_position=current_observations["fancy_cube"]["goal_position"],
            current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
        )

        # apply controler decision to manipulator robot
        self._franka.apply_action(actions)

        # pause once goal was archived
        if self._controller.is_done():
            print(PRINT_PREFIX, "Reached goal after", self._tick_count, "ticks")
            self._tick_count = 0
            self._world.pause()

        return
