import mujoco
# import math
from mujoco import viewer
import numpy as np


ball = 'ball-root'

class Scene:
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        # Get the velocity address for the ball's freejoint
        # A freejoint has 6 velocity values (3 linear, 3 angular)
        self.ball_vel_adr = self.model.joint(ball).dofadr[0]

    def movement_callback(self, model, data):
        """This function is called by MuJoCo on every step."""
        # Get the ball's linear velocity (first 3 values)
        ball_velocity = data.qvel[self.ball_vel_adr : self.ball_vel_adr+3]
        # Check if the ball's speed is above a small threshold
        speed = np.linalg.norm(ball_velocity)
        print(f"Ball speed {speed:.2f} m/s")

    def launch(self):
       mujoco.set_mjcb_control(self.movement_callback)
       # qpos_address = self.model.joint(ball).qposadr[0]
       with viewer.launch(self.model, self.data) as view:
            while view.is_running() and self.data.time < 10000.0:
                # Calculate a new position (e.g., circular path)
                # radius = 3.0
                # speed = 1.5
                # (x, y) = radius * math.cos(speed * self.data.time), radius * math.sin(speed * self.data.time)
                # Set the ball's (x, y, z) position in qpos
                # self.data.qpos[qpos_address:qpos_address+3] = [x, y, 0.11]

                mujoco.mj_step(self.model, self.data)
                view.sync()


field = Scene("xml/adult_field.xml")
# goal = Scene("xml/goal.xml")

field.launch()
