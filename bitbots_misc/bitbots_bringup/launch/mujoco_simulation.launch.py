from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    """Launch MuJoCo simulation with domain bridge for multi-robot support."""

    package_share = get_package_share_directory("bitbots_mujoco_sim")
    bridge_config = Path(package_share) / "config" / "domain_bridges" / "multi_robot_bridge.yaml"

    return LaunchDescription(
        [
            # Start the MuJoCo simulation node (domain 63)
            Node(
                package="bitbots_mujoco_sim",
                executable="sim",
                name="sim_interface",
                output="screen",
                emulate_tty=True,
            ),
            # Wait for simulation to generate config, then start domain bridge
            TimerAction(
                period=2.0,  # Wait 2 seconds for sim to generate config
                actions=[
                    LogInfo(msg=f"Starting domain bridge with config: {bridge_config}"),
                    Node(
                        package="domain_bridge",
                        executable="domain_bridge",
                        name="domain_bridge",
                        arguments=[str(bridge_config)],
                        output="screen",
                        emulate_tty=True,
                    ),
                ],
            ),
        ]
    )
