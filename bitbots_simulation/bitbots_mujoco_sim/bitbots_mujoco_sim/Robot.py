import mujoco

from bitbots_mujoco_sim.Joint import Joint, RobotJoint
from bitbots_mujoco_sim.Sensor import RobotSensor


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
