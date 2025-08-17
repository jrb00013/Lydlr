# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from setuptools import find_packages, setup
import os
import sys


package_name = 'lydlr_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'opencv-python',
        'open3d',
        'librosa',
        'scikit-image',
        'scipy',
        'pyyaml'
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='AI Storage Optimization Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo = lydlr_ai.demo:main',
            'optimizer_node = lydlr_ai.optimizer_node:main',
            'synthetic_multimodal_publisher = lydlr_ai.synthetic_multimodal_publisher:main',
            'enhanced_train = lydlr_ai.enhanced_train:main',
            'collect_training_data = lydlr_ai.collect_training_data:main',
            'test_enhanced_system = lydlr_ai.test_enhanced_system:main',
        ],
    },
)
