"""Publish the autodrive vehicle description via robot_state_publisher."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('autodrive_description')
    default_xacro_path = os.path.join(pkg_share, 'urdf', 'autodrive.urdf.xacro')

    xacro_file_arg = DeclareLaunchArgument(
        'xacro_file',
        default_value=default_xacro_path,
        description='Path to the vehicle xacro file',
    )

    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('xacro_file')]), value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    return LaunchDescription([
        xacro_file_arg,
        robot_state_publisher_node,
    ])
