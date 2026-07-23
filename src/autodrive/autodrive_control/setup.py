from setuptools import find_packages, setup

package_name = 'autodrive_control'

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
    description='Tracking-error computation and LQR steering control.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_error_node = autodrive_control.nodes.tracking_error_node:main',
            'lqr_controller_node = autodrive_control.nodes.lqr_controller_node:main',
        ],
    },
)
