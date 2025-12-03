import math
import time
from functools import wraps
from pathlib import Path
from time import perf_counter

import mujoco
from ament_index_python.packages import get_package_share_directory
from mujoco import viewer
from rclpy.node import Node
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState

from bitbots_msgs.msg import JointCommand
from bitbots_mujoco_sim.robot import Robot

# Global dictionary to store performance measurements
_perf_measurements = {}
_perf_log_file = Path("/tmp/mujoco_sim_perf.log")


def perf_timer(func):
    """Decorator to measure and log function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t1_start = perf_counter()
        result = func(*args, **kwargs)
        t1_stop = perf_counter()
        elapsed_us = (t1_stop - t1_start) * 1_000_000

        # Store measurement
        func_name = func.__name__
        if func_name not in _perf_measurements:
            _perf_measurements[func_name] = []
        _perf_measurements[func_name].append(elapsed_us)

        # Calculate average
        avg_us = sum(_perf_measurements[func_name]) / len(_perf_measurements[func_name])
        num_calls = len(_perf_measurements[func_name])

        # Write all measurements to file (overwrite)
        with open(_perf_log_file, "w") as f:
            f.write("Function Performance Metrics\n")
            f.write("=" * 80 + "\n\n")
            for fname, measurements in _perf_measurements.items():
                avg = sum(measurements) / len(measurements)
                count = len(measurements)
                last = measurements[-1]
                f.write(f"{fname}:\n")
                f.write(f"  Last: {last:.2f} µs\n")
                f.write(f"  Avg:  {avg:.2f} µs\n")
                f.write(f"  Calls: {count}\n\n")

        return result

    return wrapper


class Simulation(Node):
    """Manages the MuJoCo simulation state and its components."""

    def __init__(self):
        super().__init__("sim_interface")
        package_path = get_package_share_directory("bitbots_mujoco_sim")
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(package_path + "/xml/adult_field.xml")
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.robot: Robot = Robot(self.model, self.data)

        self.time = 0.0
        self.time_message = Time(seconds=0, nanoseconds=0).to_msg()
        self.timestep = self.model.opt.timestep
        self.step_number = 0
        self.clock_publisher = self.create_publisher(Clock, "clock", 1)

        self.js_publisher = self.create_publisher(JointState, "joint_states", 1)
        self.create_subscription(JointCommand, "DynamixelController/command", self.joint_command_callback, 1)

        self.imu_frame_id = self.get_parameter_or("imu_frame", "imu_frame")
        self.imu_publisher = self.create_publisher(Imu, "imu/data_raw", 1)

        self.camera_active = True
        self.camera_optical_frame_id = self.get_parameter_or("camera_optical_frame", "camera_optical_frame")
        self.camera_publisher = self.create_publisher(Image, "camera/image_proc", 1)
        self.camera_info_publisher = self.create_publisher(CameraInfo, "camera/camera_info", 1)

        self.events = {
            "clock": {"frequency": 1, "handler": self.publish_clock_event},
            "joint_states": {"frequency": 3, "handler": self.publish_ros_joint_states_event},
            "imu": {"frequency": 3, "handler": self.publish_imu_event},
            "camera": {"frequency": 24, "handler": self.publish_camera_event},
        }

    def run(self) -> None:
        print("Starting simulation viewer...")
        with viewer.launch_passive(self.model, self.data) as view:
            while view.is_running():
                self.step()
                view.sync()

    @perf_timer
    def joint_command_callback(self, command: JointCommand) -> None:
        if len(command.positions) != 0:
            for i in range(len(command.joint_names)):
                joint = self.robot.joints.get(command.joint_names[i])
                joint.position = command.positions[i]
                # if len(command.velocities) != 0:
                #    joint.velocity = command.velocities[i]

    @perf_timer
    def step(self) -> None:
        real_start_time = time.time()
        self.step_number += 1
        self.time += self.timestep
        self.time_message = Time(seconds=int(self.time), nanoseconds=int(self.time % 1 * 1e9)).to_msg()

        mujoco.mj_step(self.model, self.data)

        for _, event_config in self.events.items():
            if self.step_number % event_config["frequency"] == 0:
                event_config["handler"]()

        real_end_time = time.time()
        time.sleep(max(0.0, self.timestep - (real_end_time - real_start_time)))

    @perf_timer
    def publish_clock_event(self) -> None:
        clock_msg = Clock()
        clock_msg.clock = self.time_message
        self.clock_publisher.publish(clock_msg)

    @perf_timer
    def publish_ros_joint_states_event(self) -> None:
        js = JointState()
        js.name = []
        js.header.stamp = self.time_message
        js.position = []
        js.effort = []
        for joint in self.robot.joints:
            js.name.append(joint.ros_name)
            js.position.append(joint.position)
            js.velocity.append(joint.velocity)
            js.effort.append(joint.effort)
        self.js_publisher.publish(js)
    @perf_timer

    def publish_imu_event(self) -> None:
        imu = Imu()
        imu.header.stamp = self.time_message
        imu.header.frame_id = self.imu_frame_id
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (
            self.robot.sensors.accelerometer.data
        )

        # make sure that acceleration is not completely zero or we will get error in filter.
        # Happens if robot is moved manually in the simulation
        if imu.linear_acceleration.x == 0 and imu.linear_acceleration.y == 0 and imu.linear_acceleration.z == 0:
            imu.linear_acceleration.z = 0.001

        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = self.robot.sensors.gyro.data

        self.imu_publisher.publish(imu)

    @perf_timer
    def publish_camera_event(self) -> None:
        if not self.camera_active:
            return

        img = Image()
        img.header.stamp = self.time_message
        img.header.frame_id = self.camera_optical_frame_id
        img.encoding = "bgra8"
        img.height = self.robot.camera.height
        img.width = self.robot.camera.width
        img.step = 4 * self.robot.camera.width
        img.data = self.robot.camera.render()
        self.camera_publisher.publish(img)

        cam_info = CameraInfo()
        cam_info.header = img.header
        cam_info.height = self.robot.camera.height
        cam_info.width = self.robot.camera.width

        @staticmethod
        def focal_length_from_fov(fov: float, resolution: int) -> float:
            "Calculate focal length from field of view and resolution."
            return 0.5 * resolution * (math.cos(fov / 2) / math.sin(fov / 2))

        @staticmethod
        def h_fov_to_v_fov(h_fov: float, height: int, width: int) -> float:
            "Convert horizontal FOV to vertical FOV based on image aspect ratio."
            return 2 * math.atan(math.tan(h_fov * 0.5) * (height / width))

        camera = self.robot.camera

        h_fov = camera.fov
        v_fov = h_fov_to_v_fov(h_fov, camera.height, camera.width)

        f_x, f_y = focal_length_from_fov(h_fov, camera.width), focal_length_from_fov(v_fov, camera.height)
        cx, cy = camera.width / 2.0, camera.height / 2.0
        cam_info.k = [f_x, 0.0, cx, 0.0, f_y, cy, 0.0, 0.0, 1.0]
        cam_info.p = [f_x, 0.0, cx, 0.0, 0.0, f_y, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.camera_info_publisher.publish(cam_info)
