from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np


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


@dataclass
class RobotSensor:
    """A data class representing a single robot sensor's configuration and instance."""

    name: str
    ros_name: str
    instance: Optional[Sensor] = field(default=None, init=False)
