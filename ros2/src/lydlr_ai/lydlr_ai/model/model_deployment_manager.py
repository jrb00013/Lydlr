# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Model Deployment Manager
- Hot-swap models on running nodes
- Version management
- A/B testing
- Performance monitoring
- Automatic rollback
- Dynamic node discovery
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import time
import threading


class ModelDeploymentManager(Node):
    """Manages model deployment across edge nodes"""
    
    def __init__(self):
        super().__init__('model_deployment_manager')
        
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Track deployed models
        self.deployed_models: Dict[str, str] = {}  # node_id -> version
        self.previous_models: Dict[str, str] = {}  # For rollback
        self.node_metrics: Dict[str, deque] = {}  # Metrics history
        self.node_baseline_metrics: Dict[str, Dict] = {}  # Baseline for comparison
        
        # A/B testing
        self.ab_test_configs: Dict[str, Dict] = {}  # node_id -> {version_a, version_b, split}
        
        # Publishers for each node (created dynamically)
        self.deploy_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        
        # Dynamic node discovery
        self.discovered_nodes: set = set()
        self.node_discovery_timer = self.create_timer(5.0, self.discover_nodes)
        
        # Performance monitoring
        self.performance_thresholds = {
            'min_compression_ratio': 5.0,
            'max_latency_ms': 50.0,
            'min_quality_score': 0.7
        }
        
        # Service for deployment commands
        self.deploy_timer = self.create_timer(1.0, self.check_deployments)
        self.monitoring_timer = self.create_timer(2.0, self.monitor_performance)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.get_logger().info("🎯 Model Deployment Manager initialized")
        self.get_logger().info("   Features: A/B testing, auto-rollback, dynamic discovery")
    
    def discover_nodes(self):
        """Dynamically discover nodes by checking for metrics topics"""
        # Dynamically discover nodes from ROS2 topics
        # Check for any node_X pattern in active topics
        import subprocess
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Extract node IDs from topic names like /node_X/metrics
                import re
                topics = result.stdout.split('\n')
                discovered = set()
                for topic in topics:
                    match = re.search(r'/node_(\d+)/', topic)
                    if match:
                        discovered.add(f"node_{match.group(1)}")
                if discovered:
                    common_nodes = sorted(list(discovered))
                else:
                    # Fallback to default if no nodes discovered yet
                    common_nodes = ['node_0', 'node_1']
            else:
                common_nodes = ['node_0', 'node_1']  # Default fallback
        except Exception:
            # Fallback to default nodes if discovery fails
            common_nodes = ['node_0', 'node_1', 'node_2', 'node_3']
        
        for node_id in common_nodes:
            if node_id not in self.discovered_nodes:
                # Try to create subscriber - if it works, node exists
                try:
                    if node_id not in self.deploy_publishers:
                        self.deploy_publishers[node_id] = self.create_publisher(
                            String, f'/{node_id}/model/deploy', 10
                        )
                    
                    # Create metrics subscriber
                    self.create_subscription(
                        Float32MultiArray, f'/{node_id}/metrics',
                        lambda msg, nid=node_id: self.metrics_callback(nid, msg), 10
                    )
                    
                    self.discovered_nodes.add(node_id)
                    if node_id not in self.node_metrics:
                        self.node_metrics[node_id] = deque(maxlen=100)
                    
                    self.get_logger().info(f"🔍 Discovered node: {node_id}")
                except Exception as e:
                    pass  # Node doesn't exist yet
    
    def deploy_model(self, node_id: str, version: str, save_previous: bool = True) -> bool:
        """Deploy a model version to a specific node"""
        model_path = self.model_dir / f"compressor_v{version}.pth"
        
        if not model_path.exists():
            self.get_logger().error(f"Model v{version} not found at {model_path}")
            return False
        
        # Create publisher if needed
        if node_id not in self.deploy_publishers:
            self.deploy_publishers[node_id] = self.create_publisher(
                String, f'/{node_id}/model/deploy', 10
            )
        
        # Save previous model for rollback
        if save_previous and node_id in self.deployed_models:
            self.previous_models[node_id] = self.deployed_models[node_id]
        
        # Publish deployment command
        msg = String()
        msg.data = version
        self.deploy_publishers[node_id].publish(msg)
        
        with self.lock:
            self.deployed_models[node_id] = version
        
        # Establish baseline metrics
        if node_id in self.node_metrics and len(self.node_metrics[node_id]) > 0:
            self.node_baseline_metrics[node_id] = {
                'compression_ratio': self.node_metrics[node_id][-1].get('compression_ratio', 0),
                'latency_ms': self.node_metrics[node_id][-1].get('latency_ms', 0),
                'quality_score': self.node_metrics[node_id][-1].get('quality_score', 0),
                'timestamp': time.time()
            }
        
        self.get_logger().info(f"📤 Deployed model v{version} to {node_id}")
        
        return True
    
    def deploy_to_all_nodes(self, version: str, node_ids: List[str] = None):
        """Deploy model to all nodes"""
        if node_ids is None:
            # Discover nodes dynamically
            self.discover_nodes()
            node_ids = list(self.discovered_nodes) if self.discovered_nodes else ['node_0', 'node_1']
        
        for node_id in node_ids:
            self.deploy_model(node_id, version)
    
    def metrics_callback(self, node_id: str, msg: Float32MultiArray):
        """Receive metrics from nodes"""
        if len(msg.data) >= 5:
            with self.lock:
                if node_id not in self.node_metrics:
                    self.node_metrics[node_id] = deque(maxlen=100)
                
                metrics = {
                    'compression_ratio': float(msg.data[0]),
                    'latency_ms': float(msg.data[1]),
                    'compression_level': float(msg.data[2]),
                    'quality_score': float(msg.data[3]),
                    'bandwidth_estimate': float(msg.data[4]),
                    'timestamp': time.time()
                }
                
                self.node_metrics[node_id].append(metrics)
    
    def check_deployments(self):
        """Periodically check and manage deployments"""
        # Check for new model versions
        available_versions = self.list_available_models()
        
        # Auto-deploy latest version to nodes without models
        for node_id in self.discovered_nodes:
            if node_id not in self.deployed_models and available_versions:
                self.deploy_model(node_id, available_versions[0])
    
    def monitor_performance(self):
        """Monitor performance and trigger rollback if needed"""
        with self.lock:
            for node_id, metrics_buffer in self.node_metrics.items():
                if not metrics_buffer or node_id not in self.deployed_models:
                    continue
                
                latest = metrics_buffer[-1]
                baseline = self.node_baseline_metrics.get(node_id, {})
                
                # Check performance thresholds
                should_rollback = False
                reasons = []
                
                if latest['compression_ratio'] < self.performance_thresholds['min_compression_ratio']:
                    should_rollback = True
                    reasons.append(f"low compression ({latest['compression_ratio']:.2f}x)")
                
                if latest['latency_ms'] > self.performance_thresholds['max_latency_ms']:
                    should_rollback = True
                    reasons.append(f"high latency ({latest['latency_ms']:.2f}ms)")
                
                if latest['quality_score'] < self.performance_thresholds['min_quality_score']:
                    should_rollback = True
                    reasons.append(f"low quality ({latest['quality_score']:.3f})")
                
                # Compare to baseline
                if baseline:
                    compression_drop = (baseline.get('compression_ratio', 0) - latest['compression_ratio']) / max(baseline.get('compression_ratio', 1), 1)
                    quality_drop = baseline.get('quality_score', 0) - latest['quality_score']
                    
                    if compression_drop > 0.3:  # 30% drop
                        should_rollback = True
                        reasons.append(f"compression dropped {compression_drop*100:.1f}%")
                    
                    if quality_drop > 0.1:  # 10% drop
                        should_rollback = True
                        reasons.append(f"quality dropped {quality_drop*100:.1f}%")
                
                if should_rollback and node_id in self.previous_models:
                    self.get_logger().warn(
                        f"⚠️ Performance degradation on {node_id}: {', '.join(reasons)}"
                    )
                    self.rollback_model(node_id)
    
    def rollback_model(self, node_id: str) -> bool:
        """Rollback to previous model version"""
        if node_id not in self.previous_models:
            self.get_logger().error(f"No previous model to rollback for {node_id}")
            return False
        
        previous_version = self.previous_models[node_id]
        self.get_logger().info(f"🔄 Rolling back {node_id} to v{previous_version}")
        
        return self.deploy_model(node_id, previous_version, save_previous=False)
    
    def setup_ab_test(self, node_id_a: str, node_id_b: str, version_a: str, version_b: str):
        """Set up A/B testing between two nodes"""
        self.ab_test_configs[node_id_a] = {
            'version': version_a,
            'test_group': 'A'
        }
        self.ab_test_configs[node_id_b] = {
            'version': version_b,
            'test_group': 'B'
        }
        
        self.deploy_model(node_id_a, version_a)
        self.deploy_model(node_id_b, version_b)
        
        self.get_logger().info(
            f"🧪 A/B test setup: {node_id_a} (v{version_a}) vs {node_id_b} (v{version_b})"
        )
    
    def get_ab_test_results(self) -> Dict:
        """Get A/B test comparison results"""
        results = {}
        
        for node_id, config in self.ab_test_configs.items():
            if node_id in self.node_metrics and len(self.node_metrics[node_id]) > 0:
                latest = self.node_metrics[node_id][-1]
                results[node_id] = {
                    'version': config['version'],
                    'test_group': config['test_group'],
                    'metrics': latest
                }
        
        return results
    
    def get_node_performance(self, node_id: str) -> Dict:
        """Get performance metrics for a node"""
        with self.lock:
            if node_id in self.node_metrics and len(self.node_metrics[node_id]) > 0:
                return self.node_metrics[node_id][-1]
            return {}
    
    def get_node_performance_history(self, node_id: str, window: int = 10) -> List[Dict]:
        """Get recent performance history for a node"""
        with self.lock:
            if node_id in self.node_metrics:
                return list(self.node_metrics[node_id])[-window:]
            return []
    
    def list_available_models(self) -> List[str]:
        """List all available model versions"""
        versions = []
        for f in self.model_dir.glob("compressor_v*.pth"):
            version = f.stem.split("_v")[1]
            versions.append(version)
        return sorted(versions, reverse=True)
    
    def get_deployment_status(self) -> Dict:
        """Get current deployment status across all nodes"""
        with self.lock:
            status = {
                'deployed_models': self.deployed_models.copy(),
                'discovered_nodes': list(self.discovered_nodes),
                'node_performance': {}
            }
            
            for node_id in self.discovered_nodes:
                if node_id in self.node_metrics and len(self.node_metrics[node_id]) > 0:
                    status['node_performance'][node_id] = self.node_metrics[node_id][-1]
            
            return status


def main(args=None):
    rclpy.init(args=args)
    manager = ModelDeploymentManager()
    
    # Example: Deploy latest model
    versions = manager.list_available_models()
    if versions:
        manager.deploy_to_all_nodes(versions[0])
    
    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

