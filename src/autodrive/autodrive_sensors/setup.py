from setuptools import find_packages, setup

package_name = 'autodrive_sensors'

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
    description='Hardware sensor input nodes for the autodrive vehicle.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = autodrive_sensors.nodes.camera_node:main',
            'steering_feedback_node = autodrive_sensors.nodes.steering_feedback_node:main',
            'arduino_sensor_node = autodrive_sensors.nodes.arduino_sensor_node:main',
        ],
    },
)
