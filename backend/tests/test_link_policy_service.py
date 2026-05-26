"""Tests for fleet link policy service."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.api.services.link_policy_service import build_fleet_link_policy


class FakeNodeRepo:
    def __init__(self, nodes):
        self.nodes = nodes

    async def list_all(self, **kwargs):
        return self.nodes


async def _build(nodes):
    db = MagicMock()
    from backend.api.services import link_policy_service

    original = link_policy_service.NodeRepository
    link_policy_service.NodeRepository = lambda _db: FakeNodeRepo(nodes)
    try:
        return await build_fleet_link_policy(db)
    finally:
        link_policy_service.NodeRepository = original


def test_build_fleet_link_policy_drone_and_iot():
    nodes = [
        {"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512},
        {"node_id": "iot_gateway_01", "vertical": "iot", "uplink_budget_kbps": 64},
    ]
    policy = asyncio.run(_build(nodes))
    assert policy["node_count"] == 2
    assert policy["nodes"]["node_0"]["uplink_budget_kbps"] == 512
    assert policy["nodes"]["node_0"]["vision_fps_cap"] == 15
    assert policy["nodes"]["iot_gateway_01"]["uplink_budget_kbps"] == 64
    assert policy["nodes"]["iot_gateway_01"]["vision_fps_cap"] == 2
    assert policy["nodes"]["iot_gateway_01"]["prioritize"][0] == "imu"
