#!/usr/bin/env python3
"""
Launch file for collecting training data
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'sequence_length',
            default_value='10',
            description='Number of frames to collect per sequence'
        ),
        
        DeclareLaunchArgument(
            'collection_rate',
            default_value='10.0',
            description='Data collection rate in Hz'
        ),
        
        DeclareLaunchArgument(
            'save_directory',
            default_value='~/lydlr_ws/data/training_data',
            description='Directory to save training data'
        ),
        
        # Training data collector node
        Node(
            package='lydlr_ai',
            executable='collect_training_data.py',
            name='training_data_collector',
            output='screen',
            parameters=[{
                'sequence_length': LaunchConfiguration('sequence_length'),
                'collection_rate': LaunchConfiguration('collection_rate'),
                'save_directory': LaunchConfiguration('save_directory')
            }],
            remappings=[
                ('/camera/image_raw', '/camera/image_raw'),  # Adjust topic names as needed
                ('/lidar/points', '/lidar/points'),
                ('/imu/data', '/imu/data')
            ]
        )
    ])
