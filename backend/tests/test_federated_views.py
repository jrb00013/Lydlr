"""Tests for federated fleet learning API endpoints."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.schema.documents import build_federated_round_document
from backend.api.views import federated_views as fv


class FakeInsertResult:
    inserted_id = "mongo_id_1"


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    async def to_list(self, _limit):
        return [dict(r) for r in self.rows]


class FakeFederatedCollection:
    def __init__(self):
        self.docs = []

    async def insert_one(self, doc):
        stored = dict(doc)
        self.docs.append(stored)
        return FakeInsertResult()

    def find(self, query=None):
        return FakeCursor(self.docs)

    async def find_one(self, query):
        for d in self.docs:
            if all(d.get(k) == v for k, v in query.items()):
                return dict(d)
        return None


class FakeNodesCollection:
    def __init__(self, node_ids):
        self.node_ids = set(node_ids)

    async def find_one(self, query):
        nid = query.get("node_id")
        if nid in self.node_ids:
            return {"node_id": nid}
        return None


class FakeDB:
    def __init__(self, node_ids):
        self.federated_rounds = FakeFederatedCollection()
        self.nodes = FakeNodesCollection(node_ids)

    def __getitem__(self, key):
        if key == "federated_rounds":
            return self.federated_rounds
        if key == "nodes":
            return self.nodes
        raise KeyError(key)


@pytest.fixture
def fed_db():
    return FakeDB(["node_0", "node_1"])


def _exec_list(db, method, *, data=None, query=None):
    original = fv.ensure_db_connection
    fv.ensure_db_connection = AsyncMock(return_value=db)
    async def _call():
        view = fv.FederatedRoundListView()
        request = MagicMock()
        request.data = data or {}
        request.query_params = query or {}
        return await getattr(view, method.lower())(request)
    try:
        return asyncio.run(_call())
    finally:
        fv.ensure_db_connection = original


def _exec_detail(db, round_id):
    original = fv.ensure_db_connection
    fv.ensure_db_connection = AsyncMock(return_value=db)
    async def _call():
        view = fv.FederatedRoundDetailView()
        request = MagicMock()
        return await view.get(request, round_id=round_id)
    try:
        return asyncio.run(_call())
    finally:
        fv.ensure_db_connection = original


def test_build_federated_round_document_structure():
    doc = build_federated_round_document(
        ["node_0", "node_1"],
        base_version="vv1.0",
    )
    assert doc["round_id"].startswith("fr_")
    assert doc["participant_node_ids"] == ["node_0", "node_1"]
    assert doc["base_version"] == "vv1.0"
    assert doc["status"] == "pending"
    assert doc["merged_version"] is None
    assert doc["completed_at"] is None
    assert doc["max_delta_kbps"] == 128.0
    assert doc["inference_backend"] == "onnx"
    assert "node_0" in doc["participant_status"]
    assert doc["participant_status"]["node_0"]["status"] == "pending"
    assert len(doc["participants"]) == 2
    assert doc["modality_bytes_out_total"] == 0


def test_build_federated_round_document_custom():
    doc = build_federated_round_document(
        ["n1"],
        base_version="v2",
        max_delta_kbps=256,
        inference_backend="trt",
        round_id="my_round",
    )
    assert doc["max_delta_kbps"] == 256.0
    assert doc["inference_backend"] == "trt"
    assert doc["round_id"] == "fr_my_round"


def test_list_view_post_valid(fed_db):
    resp = _exec_list(fed_db, "POST", data={
        "participant_node_ids": ["node_0", "node_1"],
        "base_version": "vv1.0",
        "max_delta_kbps": 128,
    })
    assert resp.status_code == 201
    data = resp.data
    assert data["base_version"] == "vv1.0"
    assert data["status"] == "pending"
    assert len(fed_db.federated_rounds.docs) == 1


def test_list_view_post_missing_fields():
    resp = _exec_list(FakeDB(["n1"]), "POST", data={"base_version": "v1"})
    assert resp.status_code == 400


def test_list_view_post_unknown_nodes():
    resp = _exec_list(FakeDB([]), "POST", data={
        "participant_node_ids": ["nonexistent"],
        "base_version": "v1",
    })
    assert resp.status_code == 400
    assert "unknown nodes" in str(resp.data)


def test_detail_view_found(fed_db):
    create_resp = _exec_list(fed_db, "POST", data={
        "participant_node_ids": ["node_0", "node_1"],
        "base_version": "v1",
    })
    rid = create_resp.data["round_id"]
    resp = _exec_detail(fed_db, rid)
    assert resp.status_code == 200
    assert resp.data["round_id"] == rid


def test_detail_view_not_found():
    resp = _exec_detail(FakeDB([]), "fr_nonexistent")
    assert resp.status_code == 404


def test_list_get(fed_db):
    _exec_list(fed_db, "POST", data={
        "participant_node_ids": ["node_0"],
        "base_version": "v1",
    })
    _exec_list(fed_db, "POST", data={
        "participant_node_ids": ["node_1"],
        "base_version": "v2",
    })
    resp = _exec_list(fed_db, "GET")
    assert resp.status_code == 200
    assert len(resp.data) == 2
