"""Build per-node link policies from MongoDB fleet registry."""
from typing import Any, Dict, List

from backend.api.repositories.node_repository import NodeRepository


async def build_fleet_link_policy(db) -> Dict[str, Any]:
    nodes = await NodeRepository(db).list_all()
    policy_nodes: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("node_id")
        if not node_id:
            continue
        vertical = (node.get("vertical") or "drone").lower()
        budget = float(node.get("uplink_budget_kbps") or (64 if vertical == "iot" else 512))
        policy_nodes[node_id] = {
            "node_id": node_id,
            "vertical": vertical,
            "display_name": node.get("display_name", node_id),
            "uplink_budget_kbps": budget,
            "vision_fps_cap": float(
                node.get("vision_fps_cap") or (2 if vertical == "iot" else 15)
            ),
            "prioritize": node.get("prioritize") or (
                ["imu", "lidar", "camera", "audio"]
                if vertical == "iot"
                else ["lidar", "imu", "camera", "audio"]
            ),
            "min_quality": float(
                node.get("min_quality") or (0.65 if vertical == "iot" else 0.75)
            ),
            "max_latency_ms": float(node.get("max_latency_ms") or (80 if vertical == "iot" else 50)),
            "allocated_mbps": round(budget / 1000.0, 3),
        }

    return {
        "fleet_profile": "drone_iot_edge",
        "nodes": policy_nodes,
        "node_count": len(policy_nodes),
    }
