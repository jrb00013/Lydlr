#!/usr/bin/env python3
"""Seed MongoDB with drone + IoT edge fleet (safe to re-run)."""
import asyncio
import os
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb://lydlr:lydlr_password@localhost:27017/lydlr_db?authSource=admin",
)


async def seed():
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client.lydlr_db
    now = datetime.utcnow()

    nodes = [
        {
            "node_id": "node_0",
            "status": "active",
            "vertical": "drone",
            "display_name": "UAV Alpha",
            "model_version": "vv1.0",
            "uplink_budget_kbps": 512,
            "sensors": ["camera", "lidar", "imu", "audio"],
            "last_update": now,
        },
        {
            "node_id": "node_1",
            "status": "active",
            "vertical": "drone",
            "display_name": "UAV Bravo",
            "model_version": "vv1.0",
            "uplink_budget_kbps": 512,
            "sensors": ["camera", "lidar", "imu", "audio"],
            "last_update": now,
        },
        {
            "node_id": "iot_gateway_01",
            "status": "active",
            "vertical": "iot",
            "display_name": "Field Gateway 01",
            "model_version": "vv1.0",
            "uplink_budget_kbps": 64,
            "sensors": ["camera", "imu", "lidar"],
            "last_update": now,
        },
    ]
    for node in nodes:
        await db.nodes.update_one({"node_id": node["node_id"]}, {"$set": node}, upsert=True)

    devices = [
        {
            "device_id": "uav_alpha_fc",
            "device_type": "flight_controller",
            "node_id": "node_0",
            "status": "online",
            "vertical": "drone",
            "description": "Primary UAV — long-range video + LiDAR downlink",
            "last_update": now,
        },
        {
            "device_id": "uav_bravo_fc",
            "device_type": "flight_controller",
            "node_id": "node_1",
            "status": "online",
            "vertical": "drone",
            "description": "Secondary UAV — formation telemetry",
            "last_update": now,
        },
        {
            "device_id": "iot_gw_01",
            "device_type": "edge_gateway",
            "node_id": "iot_gateway_01",
            "status": "online",
            "vertical": "iot",
            "description": "Solar edge gateway — LPWAN uplink",
            "last_update": now,
        },
    ]
    for dev in devices:
        await db.devices.update_one({"device_id": dev["device_id"]}, {"$set": dev}, upsert=True)

    await db.system_config.update_one(
        {"type": "node_configuration"},
        {
            "$set": {
                "type": "node_configuration",
                "target_node_count": 3,
                "vertical": "drone",
                "fleet_profile": "drone_iot_edge",
                "auto_scale": False,
                "min_nodes": 2,
                "max_nodes": 10,
                "updated_at": now,
            }
        },
        upsert=True,
    )
    await db.system_config.update_one(
        {"type": "fleet_profile"},
        {
            "$set": {
                "type": "fleet_profile",
                "name": "drone_iot_edge",
                "description": "Dual-UAV + IoT gateway — bandwidth-adaptive compression",
                "verticals": ["drone", "iot"],
                "default_uplink_kbps": {"drone": 512, "iot": 64},
                "updated_at": now,
            }
        },
        upsert=True,
    )
    client.close()
    print("✅ Drone / IoT edge fleet seeded")


if __name__ == "__main__":
    asyncio.run(seed())
