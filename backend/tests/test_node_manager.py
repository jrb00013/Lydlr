"""Unit tests for ROS deploy helpers (mocked subprocess/docker)."""
from unittest.mock import MagicMock, patch

import pytest

from backend.api import node_manager


def test_is_ros2_fleet_node_running_false_without_docker():
    with patch.object(node_manager, "is_docker_available", return_value=False):
        assert node_manager.is_ros2_fleet_node_running("node_0") is False


def test_is_ros2_fleet_node_running_true_when_topics_found():
    mock_result = MagicMock(returncode=0, stdout="/lydlr/node_0/transport/metrics\n")
    with patch.object(node_manager, "is_docker_available", return_value=True), patch.object(
        node_manager, "get_ros2_container", return_value="lydlr-ros2"
    ), patch.object(node_manager, "get_ros2_distro", return_value="humble"), patch.object(
        node_manager.subprocess, "run", return_value=mock_result
    ):
        assert node_manager.is_ros2_fleet_node_running("node_0") is True


def test_deploy_model_to_node_tries_lydlr_topic_first():
    ok = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(node_manager, "get_ros2_distro", return_value="humble"), patch.object(
        node_manager, "is_docker_available", return_value=True
    ), patch.object(node_manager, "get_ros2_container", return_value="lydlr-ros2"), patch.object(
        node_manager.subprocess, "run", return_value=ok
    ) as mock_run:
        assert node_manager.deploy_model_to_node("node_0", "vv1.0") is True
        cmd = mock_run.call_args[0][0]
        assert "docker" in cmd
        assert "/lydlr/node_0/command/deploy" in cmd[5]


def test_deploy_model_to_node_fails_without_ros():
    with patch.object(node_manager, "get_ros2_distro", return_value=None):
        assert node_manager.deploy_model_to_node("node_0", "vv1.0") is False
