from setuptools import find_packages, setup

package_name = 'autodrive_perception'

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
    description='BEV transform and local map construction from the front camera.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_node = autodrive_perception.nodes.bev_node:main',
            'local_map_node = autodrive_perception.nodes.local_map_node:main',
            'lane_detector_node = autodrive_perception.nodes.lane_detector_node:main',
            'scan_cluster_node = autodrive_perception.nodes.scan_cluster_node:main',
            'lane_seg_node = autodrive_perception.nodes.lane_seg_node:main',
            'lane_curve_node = autodrive_perception.nodes.lane_curve_node:main',
        ],
    },
)
