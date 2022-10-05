#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController

class FrankaExample:
    def __init__(self):
        
        # setup sim
        self.setup_world()

        # simulate world
        for _ in range(1):
            self._world.step(render=True) # execute one physics step and one rendering step

        simulation_app.close() # close Isaac Sim
        return

    def setup_world(self):
        self._world = World()
        self._world.scene.add_default_ground_plane()

        # Robot specific class that provides extra functionalities
        # such as having gripper and end_effector instances.
        self._franka = self._world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))

        # add a cube for franka to pick up
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0.3, 0.3, 0.3]),
                size=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 0, 1.0]),
            )
        )
        
        # add physics step callback
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()
        print(current_observations, "<- HERE!")
        return
    
FrankaExample()