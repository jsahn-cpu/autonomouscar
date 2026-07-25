"""Bring up the perception stack.

lane_detector_node is the actually-implemented, actively-developed piece
right now (camera -> lane detection/tracking -> image-space reference
path, see autodrive_perception/core/). bev_node/local_map_node remain
unimplemented TODO skeletons (see their own docstrings) kept for a
possible longer-term full localization/mapping vision -- they're launched
too since they're harmless no-ops, but current work does not depend on
them.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    bev_params = os.path.join(bringup_share, 'config', 'bev.yaml')
    lane_detector_params = os.path.join(bringup_share, 'config', 'lane_detector.yaml')

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
    lane_detector_node = Node(
        package='autodrive_perception',
        executable='lane_detector_node',
        name='lane_detector_node',
        output='screen',
        parameters=[lane_detector_params],
    )

    return LaunchDescription([
        bev_node,
        local_map_node,
        lane_detector_node,
    ])
