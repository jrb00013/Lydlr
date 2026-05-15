"""
Drone / IoT fleet launch — full communication + transport stack.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _edge_group(node_id: str):
    vertical = "iot" if node_id.startswith("iot_") else "drone"
    return GroupAction([
        SetEnvironmentVariable("NODE_ID", node_id),
        SetEnvironmentVariable("NODE_VERTICAL", vertical),
        SetEnvironmentVariable("LYDLR_VERTICAL", vertical),
        Node(
            package="lydlr_ai",
            executable="edge_compressor_node",
            name=f"edge_{node_id}",
            output="screen",
        ),
    ])


def generate_launch_description():
    node_ids = [n.strip() for n in os.getenv("NODE_IDS", "node_0,node_1,iot_gateway_01").split(",") if n.strip()]

    actions = [
        DeclareLaunchArgument("ground_uplink_mbps", default_value="2.0"),
        SetEnvironmentVariable("LYDLR_VERTICAL", os.getenv("LYDLR_VERTICAL", "drone")),
        SetEnvironmentVariable("GROUND_UPLINK_MBPS", LaunchConfiguration("ground_uplink_mbps")),
        SetEnvironmentVariable("METRICS_API_URL", "http://127.0.0.1:8000/api/metrics/"),
        Node(package="lydlr_ai", executable="synthetic_multimodal_publisher", output="screen"),
        Node(package="lydlr_ai", executable="communication_hub", output="screen"),
        Node(package="lydlr_ai", executable="distributed_coordinator", output="screen"),
        Node(package="lydlr_ai", executable="model_deployment_manager", output="screen"),
        Node(package="lydlr_ai", executable="transport_relay", output="screen"),
    ]
    for nid in node_ids:
        actions.append(_edge_group(nid))

    return LaunchDescription(actions)
