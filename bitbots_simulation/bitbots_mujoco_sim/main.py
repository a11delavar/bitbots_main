import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np
from mujoco import viewer
from rclpy.node import Node as RclpyNode
from rclpy.time import Time
from sensor_msgs.msg import JointState


class Sensor:
    """Represents a single sensor, providing a clean interface to its value."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str):
        """
        Initializes the Sensor by finding its ID once.

        Args:
            model: The static MuJoCo MjModel object.
            data: The dynamic MuJoCo MjData object.
            sensor_name: The name of the sensor in the MJCF model's <sensor> section.
        """
        self._data: mujoco.MjData = data
        self._sensor_id: int = model.sensor(sensor_name).id

    @property
    def value(self) -> np.ndarray:
        """Gets the current sensor reading as a NumPy array."""
        return self._data.sensor[self._sensor_id].data


class Joint:
    """Represents a single controllable joint and associated actuator in the MuJoCo simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_name: str,
        actuator_name: str,
    ):
        """
        Initializes the Joint by finding and storing all necessary IDs and addresses once.

        Args:
            model: The static MuJoCo MjModel object.
            data: The dynamic MuJoCo MjData object.
            joint_name: The name of the joint in the MJCF model.
            actuator_name: The name of the corresponding actuator in the MJCF model.
        """
        self._data: mujoco.MjData = data

        self._joint_id: int = model.joint(joint_name).id
        self._qpos_addr: int = model.joint(joint_name).qposadr[0]
        self._qvel_addr: int = model.joint(joint_name).dofadr[0]
        self._actuator_id: int = model.actuator(actuator_name).id

    @property
    def position(self) -> float:
        """Gets the current joint position (angle) in radians."""
        return self._data.qpos[self._qpos_addr]

    @property
    def actuator_id(self) -> int:
        """Gets the actuator ID for this joint."""
        return self._actuator_id

    @property
    def velocity(self) -> float:
        """Gets the current joint velocity in rad/s."""
        return self._data.qvel[self._qvel_addr]

    # TODO: How to get sensor associated with this joint?
    @property
    def sensor(self) -> Sensor:
        """Gets the sensor associated with this joint."""
        # Assuming a sensor with the same name as the joint exists
        # TODO: NO! there is no sensor with the same name as the joint!
        return Sensor(self._data.model, self._data, sensor_name=self._data.model.joint(self._joint_id).name)

    def set_target(self, target_position: float) -> None:
        """Sets the position target for the joint's actuator."""
        self._data.ctrl[self._actuator_id] = target_position


@dataclass
class RobotJoint:
    """A data class representing a single robot joint's configuration and instance."""

    name: str
    ros_name: str
    instance: Optional[Joint] = field(default=None, init=False)


@dataclass
class RobotSensor:
    """A data class representing a single robot sensor's configuration and instance."""

    name: str
    ros_name: str
    instance: Optional[Sensor] = field(default=None, init=False)


class Robot:
    """Represents the robot, holding all its components like joints."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initializes the Robot by creating and mapping all its Joint objects.

        Args:
            model: The static MuJoCo MjModel object.
            data: The dynamic MuJoCo MjData object.
        """
        self.joints: list[RobotJoint] = [
            # --- Right Leg ---
            RobotJoint(name="LR_HR", ros_name="RHipYaw"),
            RobotJoint(name="LR_HAA", ros_name="RHipRoll"),
            RobotJoint(name="LR_HFE", ros_name="RHipPitch"),
            RobotJoint(name="LR_KFE", ros_name="RKnee"),
            RobotJoint(name="LR_FFE", ros_name="RAnklePitch"),
            RobotJoint(name="LR_FAA", ros_name="RAnkleRoll"),
            # --- Left Leg ---
            RobotJoint(name="LL_HR", ros_name="LHipYaw"),
            RobotJoint(name="LL_HAA", ros_name="LHipRoll"),
            RobotJoint(name="LL_HFE", ros_name="LHipPitch"),
            RobotJoint(name="LL_KFE", ros_name="LKnee"),
            RobotJoint(name="LL_FFE", ros_name="LAnklePitch"),
            RobotJoint(name="LL_FAA", ros_name="LAnkleRoll"),
            ## --- Arms (kommt noch) ---
            # RobotJoint(name="RShoulderPitch", ros_name="RShoulderPitch"),
            # RobotJoint(name="RShoulderRoll", ros_name="RShoulderRoll"),
            # RobotJoint(name="RElbow", ros_name="RElbow"),
            # RobotJoint(name="LShoulderPitch", ros_name="LShoulderPitch"),
            # RobotJoint(name="LShoulderRoll", ros_name="LShoulderRoll"),
            # RobotJoint(name="LElbow", ros_name="LElbow"),
            ## --- Head (kommt noch) ---
            # RobotJoint(name="HeadPan", ros_name="HeadPan"),
            # RobotJoint(name="HeadTilt", ros_name="HeadTilt"),
        ]
        self.sensors: list[RobotSensor] = [
            # === IMU Sensors (from your XML) ===
            RobotSensor(name="gyro", ros_name="IMU_gyro"),
            RobotSensor(name="accelerometer", ros_name="IMU_accelerometer"),
            RobotSensor(
                name="orientation",
                ros_name="IMU_orientation",
            ),  # Global orientation quaternion
            RobotSensor(name="position", ros_name="IMU_position"),  # Global position vector
            # Foot Sensos (from xml)
            RobotSensor(name="l_foot_pos", ros_name="left_foot_position"),
            RobotSensor(name="r_foot_pos", ros_name="right_foot_position"),
            RobotSensor(name="l_foot_global_linvel", ros_name="left_foot_velocity"),
            RobotSensor(name="r_foot_global_linvel", ros_name="right_foot_velocity"),
        ]
        # Populate the 'instance' field for each RobotJoint object.
        for joint in self.joints:
            joint.instance = Joint(
                model,
                data,
                joint_name=joint.name,
                actuator_name=joint.name,
            )

    def get_joint(self, name: str) -> Joint:
        """Finds and returns a specific joint by its ROS name."""
        for joint in self.joints:
            if joint.ros_name == name:
                return joint.instance  # type: ignore
        raise KeyError(f"Joint with ROS name '{name}' not found.")


class Simulation:
    """Manages the MuJoCo simulation state and its components."""

    def __init__(self, model_path: str):
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.robot: Robot = Robot(self.model, self.data)
        self.controllers = []
        self.time = 0.0
        self.ros = RclpyNode("robot_controller")
        self.publishers = {}
        self.publishers["joint_state"] = self.ros.create_publisher(JointState, "joint_states", 1)

    def add_controller(self, function) -> None:
        self.controllers.append(function)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.publish_ros_events()
        # TODO: Update all joints, sensor objects here or what was the plan?
        # TODO: handle time
        for controller in self.controllers:
            controller(self.robot, self.data)

    def publish_ros_events(self) -> None:
        self.publish_ros_joint_states_event()

    def publish_ros_joint_states_event(self) -> None:
        js = JointState()
        js.name = []
        js.header.stamp = Time(seconds=int(self.time), nanoseconds=int(self.time % 1 * 1e9)).to_msg()
        js.position = []
        js.effort = []
        for joint in self.robot.joints:
            if joint.instance is None:
                continue
            js.name.append(joint.ros_name)
            value = joint.instance.sensor.value
            js.position.append(value)
            js.velocity.append(
                joint.instance.sensor.value - value
            )  # TODO: old value - new value (not other way around)
            js.effort.append(self.data.actuator_force[joint.instance.actuator_id])
            # self.current_positions[joint.ros_name] = value # TODO: Somehow the sensor data should be updated here.
        self.publishers["joint_state"].publish(js)


simulation = Simulation("xml/adult_field.xml")


# An example controller function that makes the robot walk in place
def walk_in_place(robot, data) -> None:
    target_pos = 0.5 * math.sin(2 * data.time)
    robot.get_joint("RHipPitch").set_target(target_pos)
    robot.get_joint("RKnee").set_target(-target_pos)
    robot.get_joint("LHipPitch").set_target(target_pos)
    robot.get_joint("LKnee").set_target(-target_pos)


simulation.add_controller(walk_in_place)

with viewer.launch_passive(simulation.model, simulation.data) as view:
    while view.is_running():
        simulation.step()
        view.sync()
