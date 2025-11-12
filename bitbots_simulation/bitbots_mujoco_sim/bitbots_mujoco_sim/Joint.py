from dataclasses import field
from typing import Optional

import mujoco
from attr import dataclass


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

    def set_target(self, target_position: float) -> None:
        """Sets the position target for the joint's actuator."""
        self._data.ctrl[self._actuator_id] = target_position


@dataclass
class RobotJoint:
    """A data class representing a single robot joint's configuration and instance."""

    name: str
    ros_name: str
    instance: Optional[Joint] = field(default=None, init=False)
