import math
import time
import mujoco
from ament_index_python.packages import get_package_share_directory
from mujoco import viewer
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from bitbots_msgs.msg import JointCommand

from bitbots_mujoco_sim.Robot import Robot


class Simulation(Node):
    """Manages the MuJoCo simulation state and its components."""

    def __init__(self):
        super().__init__("sim_interface")
        package_path = get_package_share_directory("bitbots_mujoco_sim")
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(package_path + "/xml/adult_field.xml")
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.robot: Robot = Robot(self.model, self.data)
        self.controllers = []
        self.time = 0.0
        self.timestep = self.model.opt.timestep
        self.step_number = 0
        self.ros_events_publish_frequency = 3
        self.js_publisher = self.create_publisher(JointState, "joint_states", 1)
        self.create_subscription(JointCommand, "DynamixelController/command", self.joint_command_callback, 1)

    def add_controller(self, function) -> None:
        self.controllers.append(function)

    def joint_command_callback(self, command: JointCommand) -> None:
        if len(command.positions) != 0:
            for i in range(len(command.joint_names)):
                try:
                    joint = self.robot.get_joint(command.joint_names[i])
                    joint.set_position(command.positions[i])
                    if len(command.velocities) != 0:
                        joint.set_velocity(command.velocities[i])
                except ValueError:
                    print(f"invalid motor specified ({command.joint_names[i]})")

    def step(self) -> None:
        real_start_time = time.time()
        self.step_number += 1
        self.time += self.timestep

        mujoco.mj_step(self.model, self.data)

        if self.step_number % self.ros_events_publish_frequency == 0:
            self.publish_ros_events()

        for controller in self.controllers:
            controller(self.robot, self.data)

        real_end_time = time.time()
        time.sleep(max(0.0, self.timestep - (real_end_time - real_start_time)))

    def publish_ros_events(self) -> None:
        self.publish_ros_joint_states_event()

    def publish_ros_joint_states_event(self) -> None:
        js = JointState()
        js.name = []
        js.header.stamp = Time(seconds=int(self.time), nanoseconds=int(self.time % 1 * 1e9)).to_msg()
        js.position = []
        js.effort = []
        for joint in self.robot.joints:
            if joint is None:
                continue
            js.name.append(joint.ros_name)
            js.position.append(joint.position)
            js.velocity.append(joint.velocity)
            js.effort.append(self.data.actuator_force[joint.actuator_id])
        self.js_publisher.publish(js)

    def run(
        self,
    ) -> None:
        with viewer.launch_passive(self.model, self.data) as view:
            while view.is_running():
                self.step()
                view.sync()
