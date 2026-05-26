"""Tests for per-node link spec API."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from rest_framework.test import APIRequestFactory

from backend.api.views.node_views import NodeLinkSpecView


class FakeNodes:
    def __init__(self, node):
        self.node = node
        self.updated = None

    async def find_one(self, query):
        if query.get("node_id") == self.node.get("node_id"):
            return dict(self.node)
        return None

    async def update_one(self, query, update):
        self.updated = update
        self.node.update(update.get("$set", {}))
        return MagicMock(matched_count=1)


class FakeDB:
    def __init__(self, node):
        self.nodes = FakeNodes(node)


def test_patch_link_spec():
    factory = APIRequestFactory()
    request = factory.patch(
        "/api/nodes/node_0/link-spec/",
        {"uplink_budget_kbps": 256, "min_quality": 0.8},
        format="json",
    )
    fake_db = FakeDB({"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512})

    async def fake_policy(_db):
        return {"nodes": {"node_0": {"uplink_budget_kbps": 256}}}

    with patch(
        "backend.api.views.node_views.ensure_db_connection",
        AsyncMock(return_value=fake_db),
    ), patch(
        "backend.api.views.node_views.publish_message",
        AsyncMock(),
    ), patch(
        "backend.api.services.link_policy_service.build_fleet_link_policy",
        fake_policy,
    ):
        view = NodeLinkSpecView.as_view()
        response = asyncio.run(view(request, node_id="node_0"))

    assert response.status_code == 200
    assert fake_db.nodes.updated["$set"]["uplink_budget_kbps"] == 256
