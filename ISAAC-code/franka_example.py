#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
from omni.isaac.franka import Franka

world = World()
world.scene.add_default_ground_plane()
# Robot specific class that provides extra functionalities
# such as having gripper and end_effector instances.
franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
# add a cube for franka to pick up
world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0.3, 0.3, 0.3]),
        size=np.array([0.0515, 0.0515, 0.0515]),
        color=np.array([0, 0, 1.0]),
    )
)

for i in range(500):
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim
