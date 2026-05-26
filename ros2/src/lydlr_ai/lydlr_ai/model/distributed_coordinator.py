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

import os
import json
import urllib.error
import urllib.request
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray
import time
from typing import Dict, Optional
from collections import deque
import threading

from lydlr_ai.communication.topics import LydlrTopics, fleet_node_ids
from lydlr_ai.communication.qos import qos_metrics, qos_coordination, qos_compressed_egress
from lydlr_ai.communication import wire
from lydlr_ai.communication.link_policy import (
    NodeLinkPolicy,
    estimate_output_kbps,
    kbps_to_mbps,
    target_compression_level,
)


class DistributedCoordinator(Node):
    """Coordinates multiple edge compression nodes"""
    
    def __init__(self):
        super().__init__('distributed_coordinator')
        
        # Node registry
        self.registered_nodes: Dict[str, Dict] = {}
        self.node_metrics: Dict[str, deque] = {}
        self.node_compression_levels: Dict[str, float] = {}
        
        # Global bandwidth management
        self.total_bandwidth = float(os.getenv("GROUND_UPLINK_MBPS", "2.0"))
        self.allocated_bandwidth: Dict[str, float] = {}
        self.link_policies: Dict[str, NodeLinkPolicy] = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        # Subscribers for node metrics
        self.metric_subscribers = {}
        self.compressed_subscribers = {}
        
        # Publishers for coordination (LYDT wire + legacy)
        self.coordination_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.coordination_wire_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self._coord_seq = 0

        self.pub_fleet_perf = self.create_publisher(
            String, LydlrTopics.COORDINATOR_PERF, qos_coordination()
        )
        
        # Timer for coordination loop
        self.coordination_timer = self.create_timer(0.5, self.coordinate_nodes)  # 2 Hz
        self.monitoring_timer = self.create_timer(1.0, self.monitor_performance)
        self.policy_timer = self.create_timer(10.0, self.refresh_link_policies)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.get_logger().info("🌐 Distributed Coordinator initialized")

    def refresh_link_policies(self):
        """Pull uplink budgets from control plane API or env JSON."""
        policies = self._fetch_link_policies()
        if not policies:
            return
        with self.lock:
            for node_id, pdata in policies.items():
                self.link_policies[node_id] = NodeLinkPolicy.from_dict(node_id, pdata)
                budget_mbps = kbps_to_mbps(self.link_policies[node_id].uplink_budget_kbps)
                self.allocated_bandwidth[node_id] = min(budget_mbps, self.total_bandwidth)
        self.get_logger().debug(f"Refreshed link policies for {len(policies)} nodes")

    def _fetch_link_policies(self) -> Dict[str, dict]:
        raw = os.getenv("FLEET_LINK_POLICY_JSON")
        if raw:
            try:
                payload = json.loads(raw)
                return payload.get("nodes", payload)
            except json.JSONDecodeError:
                pass

        api = os.getenv("LYDLR_API_URL", "http://127.0.0.1:8000").rstrip("/")
        url = f"{api}/api/fleet/link-policy/"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = json.loads(resp.read().decode())
                return body.get("nodes", {})
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.get_logger().debug(f"Link policy fetch skipped: {exc}")
            return {}
    
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
                UInt8MultiArray,
                LydlrTopics.metrics_transport(node_id),
                lambda msg, nid=node_id: self._wire_metrics_callback(nid, msg),
                qos_metrics(),
            )
            self.create_subscription(
                Float32MultiArray,
                LydlrTopics.legacy_metrics(node_id),
                lambda msg, nid=node_id: self.node_metrics_callback(nid, msg),
                qos_metrics(),
            )

            self.compressed_subscribers[node_id] = self.create_subscription(
                UInt8MultiArray,
                LydlrTopics.compressed_transport(node_id),
                lambda msg, nid=node_id: self.compressed_data_callback(nid, msg),
                qos_compressed_egress(),
            )

            self.coordination_publishers[node_id] = self.create_publisher(
                Float32MultiArray,
                f'/{node_id}/coordination',
                qos_coordination(),
            )
            self.coordination_wire_publishers[node_id] = self.create_publisher(
                UInt8MultiArray,
                LydlrTopics.coordination(node_id),
                qos_coordination(),
            )
            
            # Allocate bandwidth
            num_nodes = len(self.registered_nodes)
            policy = self.link_policies.get(node_id)
            if policy:
                self.allocated_bandwidth[node_id] = kbps_to_mbps(policy.uplink_budget_kbps)
            else:
                self.allocated_bandwidth[node_id] = self.total_bandwidth / max(num_nodes, 1)
            
            self.get_logger().info(f"✅ Registered node: {node_id} (Bandwidth: {self.allocated_bandwidth[node_id]:.2f} Mbps)")
    
    def _ingest_metrics(self, node_id: str, metrics: dict):
        with self.lock:
            self.node_metrics[node_id].append(metrics)
            self.node_compression_levels[node_id] = metrics['compression_level']

    def _wire_metrics_callback(self, node_id: str, msg: UInt8MultiArray):
        try:
            m = wire.decode_metrics(wire.from_uint8_array(msg.data))
            self._ingest_metrics(node_id, {
                'compression_ratio': m.compression_ratio,
                'latency_ms': m.latency_ms,
                'compression_level': m.compression_level,
                'quality_score': m.quality_score,
                'bandwidth_estimate': m.bandwidth_estimate,
                'bytes_in': m.bytes_in,
                'bytes_out': m.bytes_out,
                'timestamp': time.time(),
            })
        except Exception as exc:
            self.get_logger().debug(f"wire metrics {node_id}: {exc}")

    def node_metrics_callback(self, node_id: str, msg: Float32MultiArray):
        """Legacy Float32 metrics."""
        if len(msg.data) >= 5:
            self._ingest_metrics(node_id, {
                'compression_ratio': float(msg.data[0]),
                'latency_ms': float(msg.data[1]),
                'compression_level': float(msg.data[2]),
                'quality_score': float(msg.data[3]),
                'bandwidth_estimate': float(msg.data[4]),
                'timestamp': time.time(),
            })
    
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
        """Send coordination signal to a node based on link budget."""
        if node_id not in self.coordination_publishers:
            return

        policy = self.link_policies.get(node_id) or NodeLinkPolicy.from_dict(
            node_id,
            {"vertical": "iot" if node_id.startswith("iot_") else "drone"},
        )

        latest = {}
        with self.lock:
            buf = self.node_metrics.get(node_id)
            if buf:
                latest = buf[-1]

        est_kbps = estimate_output_kbps(int(latest.get("bytes_out", 0) or 0))
        if est_kbps <= 0 and latest.get("bandwidth_estimate"):
            est_kbps = float(latest["bandwidth_estimate"])

        target_compression = target_compression_level(
            policy,
            estimated_output_kbps=est_kbps,
            quality_score=float(latest.get("quality_score", avg_quality)),
            latency_ms=float(latest.get("latency_ms", avg_latency)),
        )

        allocated = self.allocated_bandwidth.get(
            node_id,
            kbps_to_mbps(policy.uplink_budget_kbps),
        )
        
        payload = wire.CoordinationPayload(
            target_compression=target_compression,
            allocated_mbps=allocated,
            fleet_avg_compression=avg_compression,
            fleet_avg_latency_ms=avg_latency,
            fleet_avg_quality=avg_quality,
        )
        self._coord_seq += 1
        packed = wire.encode_coordination(node_id, payload, seq=self._coord_seq)
        wmsg = UInt8MultiArray()
        wmsg.data = wire.to_uint8_array_bytes(packed)
        if node_id in self.coordination_wire_publishers:
            self.coordination_wire_publishers[node_id].publish(wmsg)

        msg = Float32MultiArray()
        msg.data = [
            target_compression,
            allocated,
            avg_compression,
            avg_latency,
            avg_quality,
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
    
    # Auto-discover and register nodes dynamically
    # Nodes will be registered as they publish metrics
    # Optionally, register nodes from environment variable
    for node_id in fleet_node_ids():
        vertical = "iot" if node_id.startswith("iot_") else "drone"
        coordinator.link_policies[node_id] = NodeLinkPolicy.from_dict(
            node_id, {"vertical": vertical}
        )
        coordinator.register_node(node_id)
    coordinator.refresh_link_policies()
    # Otherwise, nodes will be auto-discovered via metrics topics
    
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

