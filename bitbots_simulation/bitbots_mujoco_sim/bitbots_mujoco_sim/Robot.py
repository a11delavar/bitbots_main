import mujoco

from bitbots_mujoco_sim.Joint import Joint
from bitbots_mujoco_sim.Sensor import Sensor


class Robot:
    """Represents the robot, holding all its components like joints."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initializes the Robot by creating and mapping all its Joint objects.

        Args:
            model: The static MuJoCo MjModel object.
            data: The dynamic MuJoCo MjData object.
        """
        self.joints: list[Joint] = [
            # --- Right Leg ---
            Joint(model, data, name="LR_HR", ros_name="RHipYaw"),
            Joint(model, data, name="LR_HAA", ros_name="RHipRoll"),
            Joint(model, data, name="LR_HFE", ros_name="RHipPitch"),
            Joint(model, data, name="LR_KFE", ros_name="RKnee"),
            Joint(model, data, name="LR_FFE", ros_name="RAnklePitch"),
            Joint(model, data, name="LR_FAA", ros_name="RAnkleRoll"),
            # --- Left Leg ---
            Joint(model, data, name="LL_HR", ros_name="LHipYaw"),
            Joint(model, data, name="LL_HAA", ros_name="LHipRoll"),
            Joint(model, data, name="LL_HFE", ros_name="LHipPitch"),
            Joint(model, data, name="LL_KFE", ros_name="LKnee"),
            Joint(model, data, name="LL_FFE", ros_name="LAnklePitch"),
            Joint(model, data, name="LL_FAA", ros_name="LAnkleRoll"),
            ## --- Arms (kommt noch) ---
            # Joint(model, data, name="RShoulderPitch", ros_name="RShoulderPitch"),
            # Joint(model, data, name="RShoulderRoll", ros_name="RShoulderRoll"),
            # Joint(model, data, name="RElbow", ros_name="RElbow"),
            # Joint(model, data, name="LShoulderPitch", ros_name="LShoulderPitch"),
            # Joint(model, data, name="LShoulderRoll", ros_name="LShoulderRoll"),
            # Joint(model, data, name="LElbow", ros_name="LElbow"),
            ## --- Head (kommt noch) ---
            # Joint(model, data, name="HeadPan", ros_name="HeadPan"),
            # Joint(model, data, name="HeadTilt", ros_name="HeadTilt"),
        ]
        self.sensors: list[Sensor] = [
            # === IMU Sensors (from your XML) ===
            Sensor(model, data, name="gyro", ros_name="IMU_gyro"),
            Sensor(model, data, name="accelerometer", ros_name="IMU_accelerometer"),
            Sensor(model, data, name="orientation", ros_name="IMU_orientation", ), # Global orientation quaternion
            Sensor(model, data, name="position", ros_name="IMU_position"),  # Global position vector
            # Foot Sensos (from xml)
            Sensor(model, data, name="l_foot_pos", ros_name="left_foot_position"),
            Sensor(model, data, name="r_foot_pos", ros_name="right_foot_position"),
            Sensor(model, data, name="l_foot_global_linvel", ros_name="left_foot_velocity"),
            Sensor(model, data, name="r_foot_global_linvel", ros_name="right_foot_velocity"),
        ]

    def get_joint(self, name: str) -> Joint:
        """Finds and returns a specific joint by its ROS name."""
        return next(filter(lambda joint: joint.ros_name == name, self.joints))
    
    def get_sensor(self, name: str) -> Sensor:
        """Finds and returns a specific sensor by its ROS name."""
        return next(filter(lambda sensor: sensor.ros_name == name, self.sensors))
