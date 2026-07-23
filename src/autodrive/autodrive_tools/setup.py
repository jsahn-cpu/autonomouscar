from setuptools import find_packages, setup

package_name = 'autodrive_tools'

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
    description='Development and testing tools for the autodrive stack.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_monitor_node = autodrive_tools.sensor_monitor_node:main',
            'dummy_pose_publisher = autodrive_tools.dummy_pose_publisher:main',
            'dummy_path_publisher = autodrive_tools.dummy_path_publisher:main',
            'camera_calibration = autodrive_tools.calibration.camera_calibration:main',
            'steering_calibration = autodrive_tools.calibration.steering_calibration:main',
        ],
    },
)
