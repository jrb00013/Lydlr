"""Unit tests for deployment serializers."""
import pytest
from datetime import datetime
from rest_framework.exceptions import ValidationError

from backend.api.serializers import (
    CompressionMetricsSerializer,
    DeploymentRequestSerializer,
    ModelRollbackSerializer,
)


def test_deployment_request_valid():
    ser = DeploymentRequestSerializer(
        data={"model_version": "vv1.0", "node_ids": ["node_0", "node_1"]}
    )
    assert ser.is_valid(), ser.errors
    assert ser.validated_data["model_version"] == "vv1.0"
    assert ser.validated_data["node_ids"] == ["node_0", "node_1"]
    assert ser.validated_data["inference_backend"] == "torch"


def test_deployment_request_onnx_backend():
    ser = DeploymentRequestSerializer(
        data={
            "model_version": "vv1.0",
            "node_ids": ["node_0"],
            "inference_backend": "onnx",
            "strategy": "canary",
        }
    )
    assert ser.is_valid(), ser.errors
    assert ser.validated_data["inference_backend"] == "onnx"
    assert ser.validated_data["strategy"] == "canary"


def test_deployment_request_missing_fields():
    ser = DeploymentRequestSerializer(data={"model_version": "vv1.0"})
    assert not ser.is_valid()
    assert "node_ids" in ser.errors


def test_rollback_serializer_empty_ok():
    ser = ModelRollbackSerializer(data={})
    assert ser.is_valid(), ser.errors
    assert ser.validated_data.get("node_ids") is None


def test_rollback_serializer_with_nodes():
    ser = ModelRollbackSerializer(data={"node_ids": ["node_0"]})
    assert ser.is_valid(), ser.errors
    assert ser.validated_data["node_ids"] == ["node_0"]


def test_compression_metrics_modality_fields():
    ser = CompressionMetricsSerializer(
        data={
            "node_id": "node_0",
            "compression_ratio": 8.5,
            "latency_ms": 12.0,
            "quality_score": 0.82,
            "bytes_in": 10000,
            "bytes_out": 1200,
            "modality_bytes_in": {"camera": 8000, "lidar": 2000},
            "modality_bytes_out": {"camera": 900, "lidar": 300},
            "modality_quality": {"camera": 0.85, "lidar": 0.78},
        }
    )
    assert ser.is_valid(), ser.errors
    assert ser.validated_data["modality_bytes_in"]["camera"] == 8000
