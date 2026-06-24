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
from pathlib import Path
import os
import sys

package_name = 'lydlr_ai'
repo_launch = Path(__file__).resolve().parents[3] / 'launch'
launch_install = []
if repo_launch.exists():
    for lf in repo_launch.glob('*.py'):
        launch_install.append((f'share/{package_name}/launch', [str(lf)]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *launch_install,
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
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'demo = lydlr_ai.demo:main',
            'optimizer_node = lydlr_ai.optimizer_node:main',
            'synthetic_multimodal_publisher = lydlr_ai.synthetic_multimodal_publisher:main',
            'enhanced_train = lydlr_ai.enhanced_train:main',
            'collect_training_data = lydlr_ai.collect_training_data:main',
            'test_enhanced_system = lydlr_ai.test_enhanced_system:main',
            'edge_compressor_node = lydlr_ai.model.edge_compressor_node:main',
            'distributed_coordinator = lydlr_ai.model.distributed_coordinator:main',
            'model_deployment_manager = lydlr_ai.model.model_deployment_manager:main',
            'train_synthetic_models = lydlr_ai.model.train_synthetic_models:main',
            'transport_relay = lydlr_ai.transport_relay_node:main',
            'communication_hub = lydlr_ai.communication_hub_node:main',
            'sensor_ingest = lydlr_ai.ingest.sensor_ingest_node:main',
            'qos_auto_tuner = lydlr_ai.model.qos_auto_tuner:main',
            'trt_inference_node = lydlr_ai.model.trt_inference_node:main',
            'async_stream_splitter = lydlr_ai.model.async_stream_splitter:main',
        ],
    },
)
