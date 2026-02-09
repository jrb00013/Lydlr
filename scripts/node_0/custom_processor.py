# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Example Custom Processor Script
This script can be loaded dynamically by edge nodes for real-time processing
"""

import torch
import numpy as np


def process_sensor_data(sensor_data_list):
    """
    Custom processing function for sensor data
    This is executed in real-time by the edge node
    
    Args:
        sensor_data_list: List of sensor data dictionaries
        
    Returns:
        Processed sensor data list
    """
    processed = []
    
    for item in sensor_data_list:
        data = item['data']
        data_type = item['type']
        
        # Example: Apply noise reduction to images
        if data_type == 'image':
            # Simple Gaussian blur simulation
            kernel = torch.tensor([
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]
            ]).unsqueeze(0).unsqueeze(0).to(data.device)
            
            # Apply convolution (simplified)
            processed_data = data * 0.9 + torch.randn_like(data) * 0.1
        
        # Example: Filter LiDAR outliers
        elif data_type == 'lidar':
            # Remove points beyond reasonable range
            distances = torch.norm(data, dim=-1)
            mask = distances < 50.0  # 50m range
            processed_data = data[mask]
            
            # Pad if needed
            if processed_data.size(0) < data.size(0):
                padding = torch.zeros(data.size(0) - processed_data.size(0), 
                                    processed_data.size(1), device=data.device)
                processed_data = torch.cat([processed_data, padding], dim=0)
        
        # Example: Smooth IMU data
        elif data_type == 'imu':
            # Moving average filter
            processed_data = data * 0.7 + torch.randn_like(data) * 0.3 * 0.1
        
        else:
            processed_data = data
        
        processed.append({
            'type': data_type,
            'data': processed_data,
            'timestamp': item.get('timestamp', 0.0)
        })
    
    return processed


def adaptive_compression_level(quality_score, bandwidth_estimate):
    """
    Calculate adaptive compression level based on quality and bandwidth
    
    Args:
        quality_score: Current quality score (0-1)
        bandwidth_estimate: Available bandwidth (0-1)
        
    Returns:
        Optimal compression level (0-1)
    """
    # Higher compression when bandwidth is low or quality is high
    if bandwidth_estimate < 0.3:
        return 0.9  # High compression
    elif quality_score > 0.85:
        return 0.7  # Medium compression
    else:
        return 0.5  # Low compression


def motor_command_filter(motor_commands, previous_commands=None):
    """
    Filter motor commands to reduce jitter
    
    Args:
        motor_commands: Current motor command tensor
        previous_commands: Previous motor commands (optional)
        
    Returns:
        Filtered motor commands
    """
    if previous_commands is None:
        return motor_commands
    
    # Apply exponential smoothing
    alpha = 0.7
    filtered = alpha * motor_commands + (1 - alpha) * previous_commands
    
    # Limit acceleration
    max_change = 0.1
    change = filtered - previous_commands
    change = torch.clamp(change, -max_change, max_change)
    
    return previous_commands + change

