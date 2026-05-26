"""Seed drone/IoT fleet and start ROS metrics collectors on startup."""
import asyncio
import logging
import os
from datetime import datetime, timezone

from backend.api.ros2_metrics_collector import start_metrics_collector

logger = logging.getLogger(__name__)

FLEET_NODES = [
    {
        "node_id": "node_0",
        "status": "active",
        "vertical": "drone",
        "display_name": "UAV Alpha",
        "model_version": "vv1.0",
        "uplink_budget_kbps": 512,
        "sensors": ["camera", "lidar", "imu", "audio"],
    },
    {
        "node_id": "node_1",
        "status": "active",
        "vertical": "drone",
        "display_name": "UAV Bravo",
        "model_version": "vv1.0",
        "uplink_budget_kbps": 512,
        "sensors": ["camera", "lidar", "imu", "audio"],
    },
    {
        "node_id": "iot_gateway_01",
        "status": "active",
        "vertical": "iot",
        "display_name": "Field Gateway 01",
        "model_version": "vv1.0",
        "uplink_budget_kbps": 64,
        "sensors": ["camera", "imu", "lidar"],
    },
]


async def seed_fleet_if_empty(db) -> int:
    """Upsert default UAV + IoT nodes when collection is empty."""
    count = await db.nodes.count_documents({})
    if count > 0:
        return 0

    now = datetime.now(timezone.utc)
    seeded = 0
    for node in FLEET_NODES:
        doc = {**node, "last_update": now}
        await db.nodes.update_one({"node_id": node["node_id"]}, {"$set": doc}, upsert=True)
        seeded += 1

    await db.system_config.update_one(
        {"type": "fleet_profile"},
        {
            "$set": {
                "type": "fleet_profile",
                "name": "drone_iot_edge",
                "verticals": ["drone", "iot"],
                "updated_at": now,
            }
        },
        upsert=True,
    )
    logger.info("Seeded %s fleet nodes (drone + IoT)", seeded)
    return seeded


def start_fleet_metrics_collectors(node_ids) -> None:
    api_url = os.getenv("API_URL", "http://localhost:8000")
    for node_id in node_ids:
        try:
            start_metrics_collector(node_id, api_url)
        except Exception as exc:
            logger.debug("Metrics collector for %s: %s", node_id, exc)


async def bootstrap_fleet(db) -> None:
    """Seed fleet if needed and start optional ROS topic collectors."""
    seeded = await seed_fleet_if_empty(db)
    nodes = await db.nodes.find({}, {"node_id": 1}).to_list(100)
    node_ids = [n["node_id"] for n in nodes if n.get("node_id")]
    if not node_ids:
        return

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, start_fleet_metrics_collectors, node_ids)
    if seeded:
        logger.info("Fleet bootstrap complete — %s nodes, metrics collectors started", len(node_ids))
