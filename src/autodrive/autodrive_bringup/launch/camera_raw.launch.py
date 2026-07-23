"""Bring up a single camera plus raw-image lane detection.

Intentionally does NOT include the rest of the vehicle stack (description,
localization, planning, control, safety, vehicle) -- this is for standalone
experimentation with the camera alone. BEV was dropped: it assumed a fixed
camera height/pitch, but the vehicle vibrates enough in practice that the
real pose drifts from those fixed values and the homography error is too
large to be useful. lane_detector_node works directly on the raw feed
instead (see autodrive_perception/core/lane_detector.py).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('autodrive_bringup')
    camera_params = os.path.join(bringup_share, 'config', 'camera.yaml')
    lane_detector_params = os.path.join(bringup_share, 'config', 'lane_detector.yaml')

    camera_node = Node(
        package='autodrive_sensors',
        executable='camera_node',
        name='camera_front_node',
        output='screen',
        parameters=[camera_params],
    )
    lane_detector_node = Node(
        package='autodrive_perception',
        executable='lane_detector_node',
        name='lane_detector_node',
        output='screen',
        parameters=[lane_detector_params],
    )

    return LaunchDescription([
        camera_node,
        lane_detector_node,
    ])
