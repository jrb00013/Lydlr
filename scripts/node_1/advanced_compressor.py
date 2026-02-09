# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Advanced Compression Script for Node 1
Implements specialized compression algorithms
"""

import torch
import torch.nn.functional as F
import numpy as np


def process_sensor_data(sensor_data_list):
    """
    Advanced processing with temporal prediction
    """
    processed = []
    previous_features = None
    
    for item in sensor_data_list:
        data = item['data']
        data_type = item['type']
        
        # Extract features
        if data_type == 'image':
            # Use spatial features
            features = F.adaptive_avg_pool2d(data, (8, 8)).flatten(1)
        elif data_type == 'lidar':
            # Use statistical features
            features = torch.cat([
                data.mean(dim=-1),
                data.std(dim=-1),
                data.max(dim=-1)[0],
                data.min(dim=-1)[0]
            ], dim=-1)
        else:
            features = data.flatten(1)
        
        # Temporal prediction
        if previous_features is not None:
            # Predict current from previous
            predicted = previous_features * 0.8
            residual = features - predicted
            
            # Only encode residual (delta compression)
            processed_data = residual
        else:
            processed_data = features
        
        previous_features = features
        
        processed.append({
            'type': data_type,
            'data': processed_data,
            'timestamp': item.get('timestamp', 0.0)
        })
    
    return processed


def quality_adaptive_compression(data, target_quality=0.8, current_quality=0.8):
    """
    Adaptively compress based on quality requirements
    """
    if current_quality < target_quality:
        # Increase quality - less compression
        compression_factor = 0.6
    else:
        # Can compress more
        compression_factor = 0.9
    
    # Apply compression
    if isinstance(data, torch.Tensor):
        compressed = data * compression_factor
        # Quantize
        compressed = torch.round(compressed * 127) / 127
    else:
        compressed = data * compression_factor
    
    return compressed

