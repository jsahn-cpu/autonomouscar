from setuptools import find_packages, setup

package_name = 'autodrive_safety'

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
    description='Final command arbitration and watchdog.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'command_arbiter_node = autodrive_safety.nodes.command_arbiter_node:main',
            'watchdog_node = autodrive_safety.nodes.watchdog_node:main',
        ],
    },
)
