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
        ],
    },
)
