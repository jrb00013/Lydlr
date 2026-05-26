"""Unit tests for deployment view logic (mocked DB/filesystem)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rest_framework.test import APIRequestFactory

from backend.api.views import deployment_views


class FakeInsertResult:
    inserted_id = "dep123"


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, _limit):
        return list(self.rows)


class FakeNodesCollection:
    def __init__(self, node_ids):
        self.node_ids = node_ids

    def find(self, query, projection=None):
        ids = query.get("node_id", {}).get("$in", [])
        rows = [{"node_id": nid} for nid in ids if nid in self.node_ids]
        return FakeCursor(rows)

    async def update_one(self, *_args, **_kwargs):
        return MagicMock(modified_count=1)


class FakeDeployments:
    def __init__(self):
        self.inserted = []
        self.updated = []

    async def insert_one(self, doc):
        self.inserted.append(doc)
        return FakeInsertResult()

    async def update_one(self, query, update):
        self.updated.append((query, update))

    def find(self):
        return FakeCursor([])


class FakeAssignments:
    def __init__(self, rows):
        self.rows = rows

    def find(self, *_args, **_kwargs):
        return FakeCursor(self.rows)

    async def find_one(self, query):
        for row in self.rows:
            if row["node_id"] == query.get("node_id"):
                return dict(row)
        return None

    async def update_one(self, *_args, **_kwargs):
        return MagicMock(modified_count=1)


class FakeArtifacts:
    async def find_one(self, query):
        version = query.get("version")
        if version:
            return {"artifact_id": f"multimodal_compressor_{version}"}
        return None


class FakeDB:
    def __init__(self, node_ids, assignments=None):
        self.nodes = FakeNodesCollection(node_ids)
        self.deployments = FakeDeployments()
        self.node_model_assignments = FakeAssignments(assignments or [])
        self.model_artifacts = FakeArtifacts()

    def __getitem__(self, name):
        return getattr(self, name)


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "lydlr_compressor_vv1.0.pth").write_bytes(b"fake weights")
    (tmp_path / "lydlr_compressor_vv2.0.pth").write_bytes(b"fake v2")
    return tmp_path


def test_deployment_post_success(model_dir):
    factory = APIRequestFactory()
    request = factory.post(
        "/api/deploy/",
        {"model_version": "vv1.0", "node_ids": ["node_0"]},
        format="json",
    )
    fake_db = FakeDB(["node_0"])

    with patch.object(deployment_views, "MODEL_DIR", model_dir), patch.object(
        deployment_views, "ensure_db_connection", AsyncMock(return_value=fake_db)
    ), patch.object(
        deployment_views, "deploy_model_to_node", return_value=True
    ), patch.object(
        deployment_views, "publish_message", AsyncMock()
    ):
        view = deployment_views.DeploymentView.as_view()
        response = asyncio.run(view(request))

    assert response.status_code == 200
    assert response.data["successful_nodes"] == ["node_0"]
    assert response.data["ros_deployed"] == ["node_0"]
    assert (model_dir / "node_0" / "lydlr_compressor_vv1.0.pth").exists()


def test_deployment_post_missing_node(model_dir):
    factory = APIRequestFactory()
    request = factory.post(
        "/api/deploy/",
        {"model_version": "vv1.0", "node_ids": ["missing"]},
        format="json",
    )
    fake_db = FakeDB(["node_0"])

    with patch.object(deployment_views, "MODEL_DIR", model_dir), patch.object(
        deployment_views, "ensure_db_connection", AsyncMock(return_value=fake_db)
    ):
        view = deployment_views.DeploymentView.as_view()
        response = asyncio.run(view(request))

    assert response.status_code == 404
    assert "missing" in response.data["detail"]


def test_rollback_post_success(model_dir):
    (model_dir / "lydlr_compressor_vv1.0.pth").write_bytes(b"v1")
    factory = APIRequestFactory()
    request = factory.post(
        "/api/deploy/rollback/",
        {"node_ids": ["node_0"]},
        format="json",
    )
    assignments = [
        {
            "node_id": "node_0",
            "model_version": "vv2.0",
            "previous_version": "vv1.0",
            "artifact_id": "multimodal_compressor_vv2.0",
        }
    ]
    fake_db = FakeDB(["node_0"], assignments=assignments)

    with patch.object(deployment_views, "MODEL_DIR", model_dir), patch.object(
        deployment_views, "ensure_db_connection", AsyncMock(return_value=fake_db)
    ), patch.object(
        deployment_views, "deploy_model_to_node", return_value=True
    ), patch.object(
        deployment_views, "publish_message", AsyncMock()
    ):
        view = deployment_views.ModelRollbackView.as_view()
        response = asyncio.run(view(request))

    assert response.status_code == 200
    assert response.data["rolled_back"][0]["model_version"] == "vv1.0"
    assert response.data["rolled_back"][0]["ros_notified"] is True


def test_rollback_skips_without_previous(model_dir):
    factory = APIRequestFactory()
    request = factory.post(
        "/api/deploy/rollback/",
        {"node_ids": ["node_0"]},
        format="json",
    )
    assignments = [
        {"node_id": "node_0", "model_version": "vv1.0", "artifact_id": "x"}
    ]
    fake_db = FakeDB(["node_0"], assignments=assignments)

    with patch.object(deployment_views, "MODEL_DIR", model_dir), patch.object(
        deployment_views, "ensure_db_connection", AsyncMock(return_value=fake_db)
    ), patch.object(deployment_views, "publish_message", AsyncMock()):
        view = deployment_views.ModelRollbackView.as_view()
        response = asyncio.run(view(request))

    assert response.status_code == 400
    assert response.data["skipped"][0]["reason"] == "no previous version"
