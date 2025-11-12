from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np


class Sensor:
    """Represents a single sensor, providing a clean interface to its value."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str, ros_name: str):
        """
        Initializes the Sensor by finding its ID once.

        Args:
            model: The static MuJoCo MjModel object.
            data: The dynamic MuJoCo MjData object.
            sensor_name: The name of the sensor in the MJCF model's <sensor> section.
        """
        self._ros_name: str = ros_name
        self._data: mujoco.MjData = data
        self._sensor_id: int = model.sensor(name).id

    @property
    def value(self) -> np.ndarray:
        """Gets the current sensor reading as a NumPy array."""
        return self._data.sensor[self._sensor_id].data

    @property
    def ros_name(self) -> str:
        """Gets the ROS name for this sensor."""
        return self._ros_name
