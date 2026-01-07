from pathlib import Path

import yaml


class DomainBridgeConfigGenerator:
    """Generates ROS 2 domain bridge configuration for all robots."""

    def __init__(self, robots: list):
        self.main_domain = 0
        self.robots = robots

    def generate_config_file(self, output_dir: Path) -> Path:
        """Generate a single config file for all robots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "multi_robot_bridge.yaml"
        config = self.generate_config()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return output_path

    def generate_config(self) -> dict:
        """Generate configuration with explicit from/to domains for all robots."""
        config = {"name": "multi_robot_bridge", "topics": []}

        for robot in self.robots:
            # Command topic: from main domain (0) to robot domain
            config["topics"].append(
                {
                    "topic": f"{robot.namespace}/DynamixelController/command",
                    "type": "bitbots_msgs/msg/JointCommand",
                    "from_domain": self.main_domain,
                    "to_domain": robot.domain,
                }
            )

            # Clock: from main domain (0) to robot domain
            config["topics"].append(
                {
                    "topic": "clock",
                    "type": "rosgraph_msgs/msg/Clock",
                    "from_domain": self.main_domain,
                    "to_domain": robot.domain,
                }
            )

            # Sensor topics: from robot domain back to main domain (0)
            sensor_topics = [
                (f"{robot.namespace}/joint_states", "sensor_msgs/msg/JointState"),
                (f"{robot.namespace}/imu/data_raw", "sensor_msgs/msg/Imu"),
                (f"{robot.namespace}/camera/image_proc", "sensor_msgs/msg/Image"),
                (f"{robot.namespace}/camera/camera_info", "sensor_msgs/msg/CameraInfo"),
                (f"{robot.namespace}/foot_pressure_left/raw", "bitbots_msgs/msg/FootPressure"),
                (f"{robot.namespace}/foot_pressure_right/raw", "bitbots_msgs/msg/FootPressure"),
                (f"{robot.namespace}/foot_center_of_pressure_left", "geometry_msgs/msg/PointStamped"),
                (f"{robot.namespace}/foot_center_of_pressure_right", "geometry_msgs/msg/PointStamped"),
            ]

            for topic, msg_type in sensor_topics:
                config["topics"].append(
                    {"topic": topic, "type": msg_type, "from_domain": robot.domain, "to_domain": self.main_domain}
                )

        return config
