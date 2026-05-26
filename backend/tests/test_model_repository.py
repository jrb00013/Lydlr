"""Unit tests for model assignment previous_version tracking."""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.api.repositories.model_repository import ModelArtifactRepository
from backend.api.schema.collections import COLLECTIONS


class FakeAssignments:
    def __init__(self, existing=None):
        self.existing = existing
        self.updates = []

    async def find_one(self, query):
        if self.existing and self.existing.get("node_id") == query.get("node_id"):
            return dict(self.existing)
        return None

    async def update_one(self, query, update, upsert=False):
        self.updates.append({"query": query, "update": update, "upsert": upsert})


class FakeArtifacts:
    pass


class FakeDB:
    def __init__(self, assignment=None):
        self._assignments = FakeAssignments(assignment)
        self._artifacts = FakeArtifacts()

    def __getitem__(self, name):
        if name == COLLECTIONS["NODE_MODEL_ASSIGNMENTS"]:
            return self._assignments
        if name == COLLECTIONS["MODEL_ARTIFACTS"]:
            return self._artifacts
        raise KeyError(name)


def test_assign_to_node_stores_previous_version(tmp_path):
    db = FakeDB(
        assignment={
            "node_id": "node_0",
            "model_version": "vv1.0",
            "artifact_id": "multimodal_compressor_vv1.0",
        }
    )
    repo = ModelArtifactRepository(db, tmp_path)
    asyncio.run(repo.assign_to_node("node_0", "vv2.0", "multimodal_compressor_vv2.0"))

    update = db._assignments.updates[0]["update"]["$set"]
    assert update["model_version"] == "vv2.0"
    assert update["previous_version"] == "vv1.0"


def test_assign_to_node_no_previous_on_first_assign(tmp_path):
    db = FakeDB()
    repo = ModelArtifactRepository(db, tmp_path)
    asyncio.run(repo.assign_to_node("node_0", "vv1.0", "multimodal_compressor_vv1.0"))

    update = db._assignments.updates[0]["update"]["$set"]
    assert update["model_version"] == "vv1.0"
    assert "previous_version" not in update
