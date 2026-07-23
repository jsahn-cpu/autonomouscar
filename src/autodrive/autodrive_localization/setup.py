from setuptools import find_packages, setup

package_name = 'autodrive_localization'

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
    description='Global map matching and EKF pose estimation.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_map_matcher_node = autodrive_localization.nodes.global_map_matcher_node:main',
            'ekf_localizer_node = autodrive_localization.nodes.ekf_localizer_node:main',
        ],
    },
)
