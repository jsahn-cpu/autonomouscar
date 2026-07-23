"""Bring up the control stack (tracking error + LQR)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    lqr_params = os.path.join(bringup_share, 'config', 'lqr.yaml')

    tracking_error_node = Node(
        package='autodrive_control',
        executable='tracking_error_node',
        name='tracking_error_node',
        output='screen',
    )
    lqr_controller_node = Node(
        package='autodrive_control',
        executable='lqr_controller_node',
        name='lqr_controller_node',
        output='screen',
        parameters=[lqr_params],
    )

    return LaunchDescription([
        tracking_error_node,
        lqr_controller_node,
    ])
