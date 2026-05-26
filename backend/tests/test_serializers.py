"""Unit tests for deployment serializers."""
import pytest
from rest_framework.exceptions import ValidationError

from backend.api.serializers import DeploymentRequestSerializer, ModelRollbackSerializer


def test_deployment_request_valid():
    ser = DeploymentRequestSerializer(
        data={"model_version": "vv1.0", "node_ids": ["node_0", "node_1"]}
    )
    assert ser.is_valid(), ser.errors
    assert ser.validated_data["model_version"] == "vv1.0"
    assert ser.validated_data["node_ids"] == ["node_0", "node_1"]


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
