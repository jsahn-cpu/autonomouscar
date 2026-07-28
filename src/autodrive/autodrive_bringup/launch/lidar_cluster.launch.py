"""Bring up the RPLIDAR A1 driver + scan_cluster_node + RViz, for testing
lidar vehicle/slot detection on a live scan.

    ros2 launch autodrive_bringup lidar_cluster.launch.py

Args:
    serial_port:=/dev/ttyUSB0   # RPLIDAR serial port
    rviz:=false                  # skip RViz (headless / echo markers instead)

RViz shows the raw /scan points plus the cluster markers from
scan_cluster_node (white = all clusters, green = car-sized, cyan sphere =
detected slot center). Tune scan_cluster.yaml against what you see.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    cluster_params = os.path.join(bringup_share, 'config', 'scan_cluster.yaml')

    serial_port = LaunchConfiguration('serial_port')
    use_rviz = LaunchConfiguration('rviz')

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

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen', condition=IfCondition(use_rviz),
        # no preset config -- add /scan (LaserScan) and /perception/clusters_viz
        # (MarkerArray) displays with Fixed Frame = laser.
    )

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        lidar_node,
        cluster_node,
        rviz_node,
    ])
