"""Bring up the RPLIDAR A1 driver + scan_cluster_node (drive-past car pass
counting) on a live scan.

    ros2 launch autodrive_bringup lidar_cluster.launch.py

Args:
    serial_port:=/dev/ttyUSB0   # RPLIDAR serial port

Watch the count:
    ros2 topic echo /perception/pass_count     # Int32, cars passed so far
    ros2 topic echo /perception/parking_ready  # Bool, True at target_count

Tune the detection zone (ROI) and gates in config/scan_cluster.yaml -- the
ROI must be a small strip that's EMPTY when no car is in it, or the count
never advances.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    cluster_params = os.path.join(bringup_share, 'config', 'scan_cluster.yaml')

    serial_port = LaunchConfiguration('serial_port')

    lidar_node = Node(
        package='sllidar_ros2', executable='sllidar_node', name='sllidar_node',
        output='screen',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': serial_port,
            'serial_baudrate': 115200,   # RPLIDAR A1
            'frame_id': 'laser',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'Sensitivity',
        }],
    )

    cluster_node = Node(
        package='autodrive_perception', executable='scan_cluster_node',
        name='scan_cluster_node', output='screen', parameters=[cluster_params],
    )

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0'),
        lidar_node,
        cluster_node,
    ])
