"""Bring up the localization stack (global map matching + EKF)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    localization_params = os.path.join(bringup_share, 'config', 'localization.yaml')

    global_map_matcher_node = Node(
        package='autodrive_localization',
        executable='global_map_matcher_node',
        name='global_map_matcher_node',
        output='screen',
        parameters=[localization_params],
    )
    ekf_localizer_node = Node(
        package='autodrive_localization',
        executable='ekf_localizer_node',
        name='ekf_localizer_node',
        output='screen',
        parameters=[localization_params],
    )

    return LaunchDescription([
        global_map_matcher_node,
        ekf_localizer_node,
    ])
