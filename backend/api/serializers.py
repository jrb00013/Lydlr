"""
Serializers for Django REST Framework
"""
from rest_framework import serializers
from datetime import datetime
from typing import List, Dict, Any, Optional


class NodeStatusSerializer(serializers.Serializer):
    node_id = serializers.CharField()
    status = serializers.CharField(default="unknown")
    model_version = serializers.CharField(required=False, allow_null=True)
    compression_ratio = serializers.FloatField(default=0.0)
    latency_ms = serializers.FloatField(default=0.0)
    quality_score = serializers.FloatField(default=0.0)
    bandwidth_estimate = serializers.FloatField(default=0.0)
    last_update = serializers.DateTimeField(default=datetime.utcnow)


class ModelInfoSerializer(serializers.Serializer):
    version = serializers.CharField()
    filename = serializers.CharField()
    size_mb = serializers.FloatField()
    created_at = serializers.CharField()
    metadata = serializers.DictField(default=dict)


class CompressionMetricsSerializer(serializers.Serializer):
    node_id = serializers.CharField()
    compression_ratio = serializers.FloatField()
    latency_ms = serializers.FloatField()
    quality_score = serializers.FloatField()
    bandwidth_estimate = serializers.FloatField(default=1.0)
    compression_level = serializers.FloatField(default=0.8)
    timestamp = serializers.DateTimeField(default=datetime.utcnow)


class SystemStatsSerializer(serializers.Serializer):
    total_nodes = serializers.IntegerField()
    active_nodes = serializers.IntegerField()
    avg_compression_ratio = serializers.FloatField()
    avg_latency_ms = serializers.FloatField()
    avg_quality_score = serializers.FloatField()
    timestamp = serializers.DateTimeField()


class DeploymentRequestSerializer(serializers.Serializer):
    model_version = serializers.CharField()
    node_ids = serializers.ListField(child=serializers.CharField())


class NodeConfigSerializer(serializers.Serializer):
    compression_level = serializers.FloatField(default=0.8)
    target_quality = serializers.FloatField(default=0.8)
    bandwidth_limit = serializers.FloatField(required=False, allow_null=True)
    enable_metrics = serializers.BooleanField(default=True)


class NodeCreateSerializer(serializers.Serializer):
    """Serializer for creating a new node"""
    node_id = serializers.CharField(required=False)  # Auto-generated if not provided
    node_type = serializers.CharField(default="edge_compressor")
    config = NodeConfigSerializer(required=False)


class NodeConfigurationSerializer(serializers.Serializer):
    """Serializer for system node configuration"""
    target_node_count = serializers.IntegerField(min_value=0, max_value=100)
    auto_scale = serializers.BooleanField(default=False)
    min_nodes = serializers.IntegerField(min_value=0, default=1)
    max_nodes = serializers.IntegerField(min_value=1, default=10)


class SensorSerializer(serializers.Serializer):
    """Serializer for sensor information"""
    sensor_id = serializers.CharField()
    sensor_type = serializers.CharField()  # e.g., 'camera', 'lidar', 'imu', 'gps', etc.
    device_id = serializers.CharField()
    status = serializers.CharField(default="active")
    data_rate = serializers.FloatField(required=False, allow_null=True)  # Hz
    resolution = serializers.CharField(required=False, allow_null=True)
    last_update = serializers.DateTimeField(default=datetime.utcnow)
    metadata = serializers.DictField(default=dict)


class DeviceSerializer(serializers.Serializer):
    """Serializer for device information"""
    device_id = serializers.CharField()
    device_name = serializers.CharField(required=False, allow_null=True)
    device_type = serializers.CharField()  # e.g., 'camera', 'lidar', 'imu', 'motor_controller', etc.
    node_id = serializers.CharField(required=False, allow_null=True)  # Connected node
    status = serializers.CharField(default="active")
    ip_address = serializers.CharField(required=False, allow_null=True)
    location = serializers.CharField(required=False, allow_null=True)
    created_at = serializers.DateTimeField(default=datetime.utcnow)
    last_update = serializers.DateTimeField(default=datetime.utcnow)
    sensors = SensorSerializer(many=True, required=False, default=list)
    metadata = serializers.DictField(default=dict)


class DeviceCreateSerializer(serializers.Serializer):
    """Serializer for creating a new device"""
    device_id = serializers.CharField(required=False)  # Auto-generated if not provided
    device_name = serializers.CharField(required=False, allow_null=True)
    device_type = serializers.CharField(default="camera")
    node_id = serializers.CharField(required=False, allow_null=True)
    ip_address = serializers.CharField(required=False, allow_null=True)
    location = serializers.CharField(required=False, allow_null=True)
    metadata = serializers.DictField(required=False, default=dict)