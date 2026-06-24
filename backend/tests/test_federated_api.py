"""Integration tests for federated rounds API."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.api.views.federated_views as fv


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
        self.docs: list[dict] = []

    async def insert_one(self, doc):
        stored = dict(doc)
        self.docs.append(stored)
        return FakeInsertResult()

    def find(self, query=None):
        if query is None or query == {}:
            return FakeCursor(self.docs)
        rid = query.get("round_id")
        if rid:
            return FakeCursor([d for d in self.docs if d.get("round_id") == rid])
        return FakeCursor(self.docs)

    async def find_one(self, query):
        for d in self.docs:
            if all(d.get(k) == v for k, v in query.items()):
                return dict(d)
        return None

    async def update_one(self, query, update):
        for d in self.docs:
            if all(d.get(k) == v for k, v in query.items()):
                d.update(update.get("$set", {}))
                return MagicMock(modified_count=1)
        return MagicMock(modified_count=0)


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


def _make_db(node_ids=None):
    return FakeDB(node_ids or ["node_0", "node_1", "node_2"])


def _run_list(method: str, *, data=None, query=None, db=None):
    db = db or _make_db()
    original = fv.ensure_db_connection
    fv.ensure_db_connection = AsyncMock(return_value=db)

    async def _call():
        view = fv.FederatedRoundListView()
        request = MagicMock()
        request.data = data or {}
        request.query_params = query or {}
        return await getattr(view, method.lower())(request)

    try:
        return asyncio.run(_call()), db
    finally:
        fv.ensure_db_connection = original


def _run_detail(round_id: str, db=None):
    db = db or _make_db()
    original = fv.ensure_db_connection
    fv.ensure_db_connection = AsyncMock(return_value=db)

    async def _call():
        view = fv.FederatedRoundDetailView()
        request = MagicMock()
        return await view.get(request, round_id=round_id)

    try:
        return asyncio.run(_call()), db
    finally:
        fv.ensure_db_connection = original


def test_start_round_success():
    resp, db = _run_list(
        "POST",
        data={
            "participant_node_ids": ["node_0", "node_1"],
            "base_version": "vv1.0",
            "max_delta_kbps": 128,
            "inference_backend": "onnx",
        },
    )
    assert resp.status_code == 201
    assert resp.data["round_id"].startswith("fr_")
    assert resp.data["base_version"] == "vv1.0"
    assert len(resp.data["participant_node_ids"]) == 2
    assert resp.data["inference_backend"] == "onnx"
    assert len(db.federated_rounds.docs) == 1


def test_start_round_missing_nodes():
    resp, _db = _run_list(
        "POST",
        data={
            "participant_node_ids": ["node_0", "missing_node"],
            "base_version": "vv1.0",
        },
    )
    assert resp.status_code == 400
    assert "missing" in resp.data["error"] or "missing" in resp.data


def test_start_round_validation():
    resp, _db = _run_list("POST", data={"base_version": "vv1.0"})
    assert resp.status_code == 400


def test_list_rounds():
    db = _make_db()
    _run_list(
        "POST",
        data={
            "participant_node_ids": ["node_0", "node_1"],
            "base_version": "vv1.0",
        },
        db=db,
    )
    resp, _db = _run_list("GET", db=db)
    assert resp.status_code == 200
    assert len(resp.data) == 1


def test_list_rounds_csv():
    db = _make_db()
    _run_list(
        "POST",
        data={
            "participant_node_ids": ["node_0", "node_1"],
            "base_version": "vv1.0",
        },
        db=db,
    )
    resp, _db = _run_list("GET", query={"format": "csv"}, db=db)
    assert resp.status_code == 200
    assert "round_id" in resp.content.decode("utf-8")


def test_round_detail():
    db = _make_db()
    create_resp, _db = _run_list(
        "POST",
        data={
            "participant_node_ids": ["node_0", "node_1"],
            "base_version": "vv1.0",
        },
        db=db,
    )
    rid = create_resp.data["round_id"]
    resp, _db = _run_detail(rid, db=db)
    assert resp.status_code == 200
    assert resp.data["round_id"] == rid


def test_round_detail_not_found():
    resp, _db = _run_detail("fr_nonexistent")
    assert resp.status_code == 404
