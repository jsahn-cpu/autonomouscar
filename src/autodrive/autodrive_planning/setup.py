import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'autodrive_planning'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'paths'), glob('paths/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rozen',
    maintainer_email='ahnjskevin@naver.com',
    description='Reference path publishing from pre-stored paths.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reference_path_node = autodrive_planning.nodes.reference_path_node:main',
        ],
    },
)
