from setuptools import find_packages, setup

package_name = 'autodrive_vehicle'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rozen',
    maintainer_email='ahnjskevin@naver.com',
    description='Arduino Mega bridge and low-level steering PID.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arduino_bridge_node = autodrive_vehicle.nodes.arduino_bridge_node:main',
            'steering_pid_node = autodrive_vehicle.nodes.steering_pid_node:main',
            'keyboard_teleop_node = autodrive_vehicle.nodes.keyboard_teleop_node:main',
            'lane_follow_node = autodrive_vehicle.nodes.lane_follow_node:main',
        ],
    },
)
