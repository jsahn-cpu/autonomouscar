"""Bring up the perception stack (BEV + local map)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    bev_params = os.path.join(bringup_share, 'config', 'bev.yaml')

    bev_node = Node(
        package='autodrive_perception',
        executable='bev_node',
        name='bev_node',
        output='screen',
        parameters=[bev_params],
    )
    local_map_node = Node(
        package='autodrive_perception',
        executable='local_map_node',
        name='local_map_node',
        output='screen',
        parameters=[bev_params],
    )

    return LaunchDescription([
        bev_node,
        local_map_node,
    ])
