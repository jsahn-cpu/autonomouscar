"""Bring up the full (non-mission) autodrive stack.

Includes: description, sensors, perception, localization, planning,
control, safety, vehicle.

autodrive_missions is intentionally NEVER launched from here.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    launch_dir = os.path.join(bringup_share, 'launch')
    config_dir = os.path.join(bringup_share, 'config')

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'description.launch.py')))
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'perception.launch.py')))
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'localization.launch.py')))
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'control.launch.py')))
    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'vehicle.launch.py')))

    camera_params = os.path.join(config_dir, 'camera.yaml')
    safety_params = os.path.join(config_dir, 'safety.yaml')

    # sensors (no dedicated launch file yet; declared inline)
    camera_front_node = Node(
        package='autodrive_sensors', executable='camera_node',
        name='camera_front_node', output='screen', parameters=[camera_params])
    camera_rear_node = Node(
        package='autodrive_sensors', executable='camera_node',
        name='camera_rear_node', output='screen', parameters=[camera_params])
    steering_feedback_node = Node(
        package='autodrive_sensors', executable='steering_feedback_node',
        name='steering_feedback_node', output='screen')
    arduino_sensor_node = Node(
        package='autodrive_sensors', executable='arduino_sensor_node',
        name='arduino_sensor_node', output='screen')

    # planning (no dedicated launch file yet; declared inline)
    reference_path_node = Node(
        package='autodrive_planning', executable='reference_path_node',
        name='reference_path_node', output='screen')

    # safety (no dedicated launch file yet; declared inline)
    command_arbiter_node = Node(
        package='autodrive_safety', executable='command_arbiter_node',
        name='command_arbiter_node', output='screen', parameters=[safety_params])
    watchdog_node = Node(
        package='autodrive_safety', executable='watchdog_node',
        name='watchdog_node', output='screen', parameters=[safety_params])

    return LaunchDescription([
        description_launch,
        camera_front_node,
        camera_rear_node,
        steering_feedback_node,
        arduino_sensor_node,
        perception_launch,
        localization_launch,
        reference_path_node,
        control_launch,
        command_arbiter_node,
        watchdog_node,
        vehicle_launch,
    ])
