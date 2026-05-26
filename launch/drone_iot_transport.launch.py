"""Launch with real sensor replay instead of purely synthetic publisher."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _edge_group(node_id: str):
    vertical = "iot" if node_id.startswith("iot_") else "drone"
    budget = "64" if vertical == "iot" else "512"
    return GroupAction([
        SetEnvironmentVariable("NODE_ID", node_id),
        SetEnvironmentVariable("NODE_VERTICAL", vertical),
        SetEnvironmentVariable("LYDLR_VERTICAL", vertical),
        SetEnvironmentVariable("UPLINK_BUDGET_KBPS", budget),
        Node(
            package="lydlr_ai",
            executable="edge_compressor_node",
            name=f"edge_{node_id}",
            output="screen",
        ),
    ])


def generate_launch_description():
    node_ids = [
        n.strip()
        for n in os.getenv("NODE_IDS", "node_0,node_1,iot_gateway_01").split(",")
        if n.strip()
    ]
    sensor_source = os.getenv("LYDLR_SENSOR_SOURCE", "replay")

    actions = [
        DeclareLaunchArgument("ground_uplink_mbps", default_value="2.0"),
        DeclareLaunchArgument("sensor_source", default_value=sensor_source),
        SetEnvironmentVariable("LYDLR_VERTICAL", os.getenv("LYDLR_VERTICAL", "drone")),
        SetEnvironmentVariable("GROUND_UPLINK_MBPS", LaunchConfiguration("ground_uplink_mbps")),
        SetEnvironmentVariable("LYDLR_API_URL", os.getenv("LYDLR_API_URL", "http://127.0.0.1:8000")),
        SetEnvironmentVariable("METRICS_API_URL", "http://127.0.0.1:8000/api/metrics/"),
        SetEnvironmentVariable("LYDLR_SENSOR_SOURCE", LaunchConfiguration("sensor_source")),
        Node(
            package="lydlr_ai",
            executable="sensor_ingest",
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("sensor_source"), "' != 'synthetic'"])
            ),
            output="screen",
        ),
        Node(
            package="lydlr_ai",
            executable="synthetic_multimodal_publisher",
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("sensor_source"), "' == 'synthetic'"])
            ),
            output="screen",
        ),
        Node(package="lydlr_ai", executable="communication_hub", output="screen"),
        Node(package="lydlr_ai", executable="distributed_coordinator", output="screen"),
        Node(package="lydlr_ai", executable="model_deployment_manager", output="screen"),
        Node(package="lydlr_ai", executable="transport_relay", output="screen"),
    ]
    for nid in node_ids:
        actions.append(_edge_group(nid))

    return LaunchDescription(actions)
