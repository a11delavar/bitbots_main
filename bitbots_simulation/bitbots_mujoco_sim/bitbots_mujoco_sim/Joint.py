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
        name: str,
        ros_name: str,
    ):
        """
        Initializes the Joint by finding and storing all necessary IDs and addresses once.

        Args:
            model: The static MuJoCo MjModel object.
            data: The dynamic MuJoCo MjData object.
            name: The name of the joint in the MJCF model.
            ros_name: The name of the corresponding actuator in the MJCF model.
        """
        self._data: mujoco.MjData = data
        self._joint_id: int = model.joint(name).id
        self._qpos_addr: int = model.joint(name).qposadr[0]
        self._qvel_addr: int = model.joint(name).dofadr[0]
        self._actuator_id: int = model.actuator(name).id
        self._ros_name: str = ros_name

    @property
    def position(self) -> float:
        """Gets the current joint position (angle) in radians."""
        return self._data.qpos[self._qpos_addr]

    @property
    def actuator_id(self) -> int:
        """Gets the actuator ID for this joint."""
        return self._actuator_id
    
    @property
    def ros_name(self) -> str:
        """Gets the ROS name for this joint."""
        return self._ros_name

    @property
    def velocity(self) -> float:
        """Gets the current joint velocity in rad/s."""
        return self._data.qvel[self._qvel_addr]
    
    def get_min_position(self) -> float:
        """Gets the minimum position limit for the joint."""
        return self._data.jnt_range[self._joint_id][0]

    def get_max_position(self) -> float:
        """Gets the maximum position limit for the joint."""
        return self._data.jnt_range[self._joint_id][1]

    def set_position(self, value: float) -> None:
        """Sets the position target for the joint's actuator."""
        self._data.ctrl[self._actuator_id] = value

    def get_min_velocity(self) -> float:
        """Gets the minimum velocity for the joint's actuator."""
        return self._data.actuator_gainprm[self._actuator_id][0]
    
    def get_max_velocity(self) -> float:
        """Gets the maximum velocity for the joint's actuator."""
        return self._data.actuator_gainprm[self._actuator_id][1]

    def set_velocity(self, target_velocity: float) -> None:
        """Sets the velocity target for the joint's actuator."""
        self._data.ctrl[self._actuator_id] = self.get_max_velocity() if target_velocity == -1 else target_velocity
