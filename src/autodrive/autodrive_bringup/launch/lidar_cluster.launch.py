"""Bring up the RPLIDAR A1 driver + scan_cluster_node (drive-past car pass
counting) on a live scan.

    ros2 launch autodrive_bringup lidar_cluster.launch.py

Args:
    serial_port:=/dev/ttyUSB0   # RPLIDAR serial port
    rviz:=false                  # skip RViz

RViz (Fixed Frame = laser): add /scan (LaserScan) and /perception/zone_viz
(MarkerArray). The zone box shows the detection ROI (cyan; GREEN while a car
occupies it), a green box on each car in the zone, and the running count as
text. Watch the count on /perception/pass_count too. Tune the ROI/gates in
config/scan_cluster.yaml -- the ROI must be EMPTY when no car is in it, or
the count never advances.
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
    )

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        lidar_node,
        cluster_node,
        rviz_node,
    ])
