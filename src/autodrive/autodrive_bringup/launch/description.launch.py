"""Bring up the vehicle description (delegates to autodrive_description)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description() -> LaunchDescription:
    description_share = get_package_share_directory('autodrive_description')
    description_launch = os.path.join(description_share, 'launch', 'description.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(description_launch)),
    ])
