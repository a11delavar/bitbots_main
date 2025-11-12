import math

import rclpy

from bitbots_mujoco_sim.Simulation import Simulation


# An example controller function that makes the robot walk in place
def walk_in_place(robot, data) -> None:
    target_pos = 0.5 * math.sin(2 * data.time)
    robot.get_joint("RHipPitch").set_position(target_pos)
    robot.get_joint("RKnee").set_position(-target_pos)
    robot.get_joint("LHipPitch").set_position(target_pos)
    robot.get_joint("LKnee").set_position(-target_pos)


def main(args=None):
    rclpy.init(args=args)
    simulation = Simulation()
    simulation.add_controller(walk_in_place)
    simulation.run()
    simulation.destroy_node()
    rclpy.shutdown()
