# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Distributed Coordinator Node
- Orchestrates multiple edge nodes
- Manages bandwidth allocation
- Coordinates compression strategies
- Real-time performance optimization
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import json
import time
from typing import Dict, List, Optional
from collections import deque
import threading


class DistributedCoordinator(Node):
    """Coordinates multiple edge compression nodes"""
    
    def __init__(self):
        super().__init__('distributed_coordinator')
        
        # Node registry
        self.registered_nodes: Dict[str, Dict] = {}
        self.node_metrics: Dict[str, deque] = {}
        self.node_compression_levels: Dict[str, float] = {}
        
        # Global bandwidth management
        self.total_bandwidth = 100.0  # Mbps (configurable)
        self.allocated_bandwidth: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        # Subscribers for node metrics
        self.metric_subscribers = {}
        self.compressed_subscribers = {}
        
        # Publishers for coordination
        self.coordination_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        
        # Timer for coordination loop
        self.coordination_timer = self.create_timer(0.5, self.coordinate_nodes)  # 2 Hz
        self.monitoring_timer = self.create_timer(1.0, self.monitor_performance)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.get_logger().info("🌐 Distributed Coordinator initialized")
    
    def register_node(self, node_id: str, node_type: str = "edge_compressor"):
        """Register a new edge node"""
        with self.lock:
            self.registered_nodes[node_id] = {
                'type': node_type,
                'registered_at': time.time(),
                'status': 'active'
            }
            
            # Initialize metrics buffer
            self.node_metrics[node_id] = deque(maxlen=50)
            self.node_compression_levels[node_id] = 0.8
            
            # Create subscribers for this node
            self.metric_subscribers[node_id] = self.create_subscription(
                Float32MultiArray,
                f'/{node_id}/metrics',
                lambda msg, nid=node_id: self.node_metrics_callback(nid, msg),
                10
            )
            
            self.compressed_subscribers[node_id] = self.create_subscription(
                UInt8MultiArray,
                f'/{node_id}/compressed',
                lambda msg, nid=node_id: self.compressed_data_callback(nid, msg),
                10
            )
            
            # Create coordination publisher
            self.coordination_publishers[node_id] = self.create_publisher(
                Float32MultiArray,
                f'/{node_id}/coordination',
                10
            )
            
            # Allocate bandwidth
            num_nodes = len(self.registered_nodes)
            self.allocated_bandwidth[node_id] = self.total_bandwidth / num_nodes
            
            self.get_logger().info(f"✅ Registered node: {node_id} (Bandwidth: {self.allocated_bandwidth[node_id]:.2f} Mbps)")
    
    def node_metrics_callback(self, node_id: str, msg: Float32MultiArray):
        """Receive metrics from a node"""
        if len(msg.data) >= 5:
            with self.lock:
                metrics = {
                    'compression_ratio': float(msg.data[0]),
                    'latency_ms': float(msg.data[1]),
                    'compression_level': float(msg.data[2]),
                    'quality_score': float(msg.data[3]),
                    'bandwidth_estimate': float(msg.data[4]),
                    'timestamp': time.time()
                }
                
                self.node_metrics[node_id].append(metrics)
                
                # Update compression level
                self.node_compression_levels[node_id] = metrics['compression_level']
    
    def compressed_data_callback(self, node_id: str, msg: UInt8MultiArray):
        """Receive compressed data from a node"""
        data_size = len(msg.data) / 1024.0  # KB
        
        # Track data flow
        with self.lock:
            if node_id not in self.allocated_bandwidth:
                return
            
            # Monitor bandwidth usage
            current_usage = data_size * 8 / 0.1  # Convert to Mbps (assuming 0.1s window)
            allocated = self.allocated_bandwidth[node_id]
            
            if current_usage > allocated * 1.1:  # 10% threshold
                self.get_logger().warn(
                    f"⚠️ {node_id} exceeding bandwidth: {current_usage:.2f} > {allocated:.2f} Mbps"
                )
    
    def coordinate_nodes(self):
        """Main coordination loop - optimizes compression across nodes"""
        if not self.registered_nodes:
            return
        
        with self.lock:
            # Calculate global performance metrics
            total_compression = 0.0
            total_latency = 0.0
            total_quality = 0.0
            active_nodes = 0
            
            for node_id, metrics_buffer in self.node_metrics.items():
                if metrics_buffer:
                    latest = metrics_buffer[-1]
                    total_compression += latest['compression_ratio']
                    total_latency += latest['latency_ms']
                    total_quality += latest['quality_score']
                    active_nodes += 1
            
            if active_nodes == 0:
                return
            
            avg_compression = total_compression / active_nodes
            avg_latency = total_latency / active_nodes
            avg_quality = total_quality / active_nodes
            
            # Adaptive bandwidth allocation
            self.adaptive_bandwidth_allocation()
            
            # Send coordination signals to each node
            for node_id in self.registered_nodes.keys():
                self.send_coordination_signal(node_id, avg_compression, avg_latency, avg_quality)
            
            # Log performance
            self.performance_history.append({
                'timestamp': time.time(),
                'avg_compression': avg_compression,
                'avg_latency': avg_latency,
                'avg_quality': avg_quality,
                'active_nodes': active_nodes
            })
    
    def adaptive_bandwidth_allocation(self):
        """Dynamically allocate bandwidth based on node performance"""
        if len(self.registered_nodes) < 2:
            return
        
        # Calculate performance scores for each node
        node_scores = {}
        total_score = 0.0
        
        for node_id, metrics_buffer in self.node_metrics.items():
            if metrics_buffer:
                latest = metrics_buffer[-1]
                # Score based on compression ratio, quality, and latency
                score = (
                    latest['compression_ratio'] * 0.4 +
                    latest['quality_score'] * 0.4 +
                    (100.0 / max(latest['latency_ms'], 1.0)) * 0.2
                )
                node_scores[node_id] = score
                total_score += score
        
        if total_score == 0:
            return
        
        # Allocate bandwidth proportionally
        for node_id, score in node_scores.items():
            allocation = (score / total_score) * self.total_bandwidth
            self.allocated_bandwidth[node_id] = allocation
    
    def send_coordination_signal(self, node_id: str, avg_compression: float, 
                                 avg_latency: float, avg_quality: float):
        """Send coordination signal to a node"""
        if node_id not in self.coordination_publishers:
            return
        
        # Calculate target compression level based on global performance
        target_compression = 0.8
        
        # Adjust based on latency
        if avg_latency > 50.0:  # High latency
            target_compression = min(0.95, target_compression + 0.1)
        elif avg_latency < 20.0:  # Low latency
            target_compression = max(0.5, target_compression - 0.1)
        
        # Adjust based on quality
        if avg_quality < 0.7:  # Low quality
            target_compression = max(0.6, target_compression - 0.1)
        
        # Get allocated bandwidth
        allocated = self.allocated_bandwidth.get(node_id, self.total_bandwidth / len(self.registered_nodes))
        
        # Create coordination message
        msg = Float32MultiArray()
        msg.data = [
            target_compression,  # Target compression level
            allocated,            # Allocated bandwidth (Mbps)
            avg_compression,      # Global average compression
            avg_latency,         # Global average latency
            avg_quality          # Global average quality
        ]
        
        self.coordination_publishers[node_id].publish(msg)
    
    def monitor_performance(self):
        """Monitor and log overall system performance"""
        if not self.performance_history:
            return
        
        latest = self.performance_history[-1]
        
        self.get_logger().info(
            f"📊 System Performance: "
            f"Compression={latest['avg_compression']:.2f}x | "
            f"Latency={latest['avg_latency']:.2f}ms | "
            f"Quality={latest['avg_quality']:.3f} | "
            f"Nodes={latest['active_nodes']}"
        )
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        with self.lock:
            status = {
                'registered_nodes': list(self.registered_nodes.keys()),
                'bandwidth_allocation': self.allocated_bandwidth.copy(),
                'node_metrics': {},
                'performance_history': list(self.performance_history)
            }
            
            for node_id, metrics_buffer in self.node_metrics.items():
                if metrics_buffer:
                    status['node_metrics'][node_id] = metrics_buffer[-1]
            
            return status


def main(args=None):
    rclpy.init(args=args)
    coordinator = DistributedCoordinator()
    
    # Register default nodes
    coordinator.register_node('node_0')
    coordinator.register_node('node_1')
    
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

