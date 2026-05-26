"""Unit tests for fleet bootstrap (no live MongoDB)."""
import asyncio
from unittest.mock import patch

import pytest

from backend.api.services.fleet_bootstrap import (
    FLEET_NODES,
    bootstrap_fleet,
    seed_fleet_if_empty,
)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, _limit):
        return list(self.rows)


class FakeNodesCollection:
    def __init__(self, count=0):
        self.count = count
        self.upserts = []

    async def count_documents(self, _query=None):
        return self.count

    async def update_one(self, query, update, upsert=False):
        self.upserts.append((query, update, upsert))

    def find(self, *_args, **_kwargs):
        return FakeCursor([{"node_id": n["node_id"]} for n in FLEET_NODES])


class FakeSystemConfig:
    def __init__(self):
        self.upserts = []

    async def update_one(self, query, update, upsert=False):
        self.upserts.append((query, update, upsert))


class FakeDB:
    def __init__(self, node_count=0):
        self.nodes = FakeNodesCollection(node_count)
        self.system_config = FakeSystemConfig()


@pytest.mark.parametrize("node_count,expected", [(0, 3), (5, 0)])
def test_seed_fleet_if_empty(node_count, expected):
    db = FakeDB(node_count=node_count)
    seeded = asyncio.run(seed_fleet_if_empty(db))
    assert seeded == expected
    if expected:
        assert len(db.nodes.upserts) == 3
        assert db.system_config.upserts


def test_bootstrap_fleet_starts_collectors():
    db = FakeDB(node_count=1)

    with patch(
        "backend.api.services.fleet_bootstrap.start_fleet_metrics_collectors"
    ) as mock_start:
        asyncio.run(bootstrap_fleet(db))
        mock_start.assert_called_once()
        node_ids = mock_start.call_args[0][0]
        assert "node_0" in node_ids
