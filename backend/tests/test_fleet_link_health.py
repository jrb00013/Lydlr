"""Tests for GET /api/fleet/link-policy/health/ endpoint."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run_view(nodes, metrics_map=None):
    """
    Execute FleetLinkHealthView.get with controlled test data.
    Mocks ensure_db_connection to return a fake db that responds to
    metrics.find_one and nodes.find (used by build_fleet_link_policy).
    """
    import backend.api.views.fleet_views as fv
    from backend.api.services import link_policy_service

    metrics_map = metrics_map or {}

    async def mock_metrics_find_one(filter, sort=None):
        nid = filter.get("node_id", "")
        return metrics_map.get(nid)

    class FakeCursor:
        def __init__(self, docs):
            self.docs = docs

        def sort(self, *a, **kw):
            return self

        async def to_list(self, _limit):
            return [dict(d) for d in self.docs]

    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs

        async def find_one(self, filter, sort=None):
            for d in self.docs:
                if all(d.get(k) == v for k, v in filter.items()):
                    return dict(d)
            return None

        async def count_documents(self, query=None):
            return len(self.docs)

        def find(self, *a, **kw):
            return FakeCursor(self.docs)

    class FakeDB:
        def __getitem__(self, key):
            return self.nodes if key == "nodes" else self

    db = FakeDB()
    db.metrics = AsyncMock()
    db.metrics.find_one = mock_metrics_find_one
    db.nodes = FakeCollection(nodes)
    db.system_config = MagicMock()
    db.system_config.find_one = AsyncMock(return_value=None)

    original_ensure = fv.ensure_db_connection
    fv.ensure_db_connection = AsyncMock(return_value=db)

    async def _call():
        view = fv.FleetLinkHealthView()
        request = MagicMock()
        request.query_params = {}
        return await view.get(request)

    try:
        return asyncio.run(_call())
    finally:
        fv.ensure_db_connection = original_ensure


def test_health_empty_fleet():
    resp = _run_view([], {})
    assert resp.status_code == 200
    data = resp.data
    assert data["summary"]["total"] == 0


def test_health_node_over_budget():
    """bytes_out=12800 → 1024 kbps → over_budget on 512."""
    nodes = [{"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512}]
    metrics = {
        "node_0": {
            "node_id": "node_0",
            "bytes_out": 12800,
            "quality_score": 0.80,
            "latency_ms": 20,
            "compression_ratio": 4.0,
            "vertical": "drone",
        },
    }
    resp = _run_view(nodes, metrics)
    assert resp.status_code == 200
    data = resp.data
    assert len(data["nodes"]) == 1
    node = data["nodes"][0]
    assert node["status"] == "over_budget", f"got {node}"
    assert node["estimated_throughput_kbps"] > node["uplink_budget_kbps"]
    assert data["summary"]["over_budget"] == 1


def test_health_node_under_budget():
    """bytes_out=500 → 40 kbps → under_budget."""
    nodes = [{"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512}]
    metrics = {
        "node_0": {
            "node_id": "node_0",
            "bytes_out": 500,
            "quality_score": 0.85,
            "latency_ms": 10,
            "compression_ratio": 2.0,
            "vertical": "drone",
        },
    }
    resp = _run_view(nodes, metrics)
    assert resp.status_code == 200
    data = resp.data
    assert data["nodes"][0]["status"] == "under_budget"
    assert data["summary"]["under_budget"] == 1


def test_health_near_budget():
    """bytes_out=3200 → 256 kbps → at_budget (util 0.5)."""
    nodes = [{"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512}]
    metrics = {
        "node_0": {
            "node_id": "node_0",
            "bytes_out": 3200,
            "quality_score": 0.80,
            "latency_ms": 15,
            "compression_ratio": 3.0,
            "vertical": "drone",
        },
    }
    resp = _run_view(nodes, metrics)
    data = resp.data
    # 3200*8/0.1/1000=256 kbps → utilization=0.5 → at_budget
    assert data["nodes"][0]["status"] == "at_budget", f"got {data['nodes'][0]}"
    assert data["summary"]["at_budget"] == 1


def test_health_quality_issue_detected():
    nodes = [{"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512}]
    metrics = {
        "node_0": {
            "node_id": "node_0",
            "bytes_out": 300,
            "quality_score": 0.30,
            "latency_ms": 50,
            "compression_ratio": 1.5,
            "vertical": "drone",
        },
    }
    resp = _run_view(nodes, metrics)
    data = resp.data
    assert data["nodes"][0]["quality_ok"] is False
    assert data["summary"]["nodes_with_quality_issues"] == 1


def test_health_mixed_fleet():
    """Multiple nodes with different budget statuses."""
    nodes = [
        {"node_id": "node_0", "vertical": "drone", "uplink_budget_kbps": 512},
        {"node_id": "iot_01", "vertical": "iot", "uplink_budget_kbps": 64},
    ]
    metrics = {
        "node_0": {
            "node_id": "node_0",
            "bytes_out": 12800,
            "quality_score": 0.80,
            "latency_ms": 20,
            "compression_ratio": 4.0,
            "vertical": "drone",
        },
        "iot_01": {
            "node_id": "iot_01",
            "bytes_out": 10,
            "quality_score": 0.70,
            "latency_ms": 40,
            "compression_ratio": 1.2,
            "vertical": "iot",
        },
    }
    resp = _run_view(nodes, metrics)
    data = resp.data
    statuses = {n["node_id"]: n["status"] for n in data["nodes"]}
    assert statuses["node_0"] == "over_budget"
    assert statuses["iot_01"] == "under_budget"
    assert data["summary"]["over_budget"] == 1
    assert data["summary"]["under_budget"] == 1
