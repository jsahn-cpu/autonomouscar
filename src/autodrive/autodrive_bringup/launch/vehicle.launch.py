"""Bring up the vehicle interface stack (Arduino bridge + steering PID)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    vehicle_params = os.path.join(bringup_share, 'config', 'vehicle.yaml')

    arduino_bridge_node = Node(
        package='autodrive_vehicle',
        executable='arduino_bridge_node',
        name='arduino_bridge_node',
        output='screen',
        parameters=[vehicle_params],
    )
    steering_pid_node = Node(
        package='autodrive_vehicle',
        executable='steering_pid_node',
        name='steering_pid_node',
        output='screen',
        parameters=[vehicle_params],
    )

    return LaunchDescription([
        arduino_bridge_node,
        steering_pid_node,
    ])
