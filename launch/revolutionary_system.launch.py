# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Launch file for the revolutionary compression system
Launches all components: nodes, coordinator, deployment manager
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    num_nodes_arg = DeclareLaunchArgument(
        'num_nodes',
        default_value='2',
        description='Number of edge nodes to launch'
    )
    
    start_coordinator_arg = DeclareLaunchArgument(
        'start_coordinator',
        default_value='true',
        description='Start distributed coordinator'
    )
    
    start_deployment_manager_arg = DeclareLaunchArgument(
        'start_deployment_manager',
        default_value='true',
        description='Start model deployment manager'
    )
    
    start_synthetic_publisher_arg = DeclareLaunchArgument(
        'start_synthetic_publisher',
        default_value='true',
        description='Start synthetic data publisher'
    )
    
    # Get launch arguments
    num_nodes = LaunchConfiguration('num_nodes')
    start_coordinator = LaunchConfiguration('start_coordinator')
    start_deployment_manager = LaunchConfiguration('start_deployment_manager')
    start_synthetic_publisher = LaunchConfiguration('start_synthetic_publisher')
    
    # Edge nodes
    edge_nodes = []
    for i in range(2):  # Default to 2 nodes, can be expanded
        node_id = f'node_{i}'
        edge_node = Node(
            package='lydlr_ai',
            executable='edge_compressor_node',
            name=f'edge_compressor_{i}',
            parameters=[{
                'node_id': node_id,
            }],
            environment=[
                {'NODE_ID': node_id}
            ],
            output='screen'
        )
        edge_nodes.append(edge_node)
    
    # Distributed coordinator
    coordinator_node = Node(
        package='lydlr_ai',
        executable='distributed_coordinator',
        name='distributed_coordinator',
        condition=IfCondition(start_coordinator),
        output='screen'
    )
    
    # Model deployment manager
    deployment_manager_node = Node(
        package='lydlr_ai',
        executable='model_deployment_manager',
        name='model_deployment_manager',
        condition=IfCondition(start_deployment_manager),
        output='screen'
    )
    
    # Synthetic data publisher
    synthetic_publisher_node = Node(
        package='lydlr_ai',
        executable='synthetic_multimodal_publisher',
        name='synthetic_multimodal_publisher',
        condition=IfCondition(start_synthetic_publisher),
        output='screen'
    )
    
    return LaunchDescription([
        num_nodes_arg,
        start_coordinator_arg,
        start_deployment_manager_arg,
        start_synthetic_publisher_arg,
        
        # Edge nodes
        *edge_nodes,
        
        # Coordinator and manager
        coordinator_node,
        deployment_manager_node,
        
        # Synthetic publisher
        synthetic_publisher_node,
    ])

