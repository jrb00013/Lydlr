"""
Model registry views — artifacts table, sync, detail.
"""
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, APIView, ensure_db_connection
from backend.api.serializers import (
    ModelInfoSerializer,
    ModelArtifactSerializer,
    ModelRegistryTableRowSerializer,
)
from backend.api.services.model_registry_service import ModelRegistryService

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))


class ModelListView(APIView):
    """List models from disk (legacy) + optional ?source=registry"""

    def get(self, request):
        source = request.query_params.get("source", "disk")
        if source == "registry":
            return Response({"detail": "Use async /api/models/registry/"}, status=400)

        models = []
        if MODEL_DIR.exists():
            for pth_file in MODEL_DIR.glob("*.pth"):
                version = (
                    pth_file.stem.split("_v")[1]
                    if "_v" in pth_file.stem
                    else "unknown"
                )
                metadata = {}
                for meta_path in MODEL_DIR.glob(f"metadata*{version}*.json"):
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                    break
                models.append({
                    "version": version,
                    "filename": pth_file.name,
                    "size_mb": pth_file.stat().st_size / (1024 * 1024),
                    "created_at": datetime.fromtimestamp(
                        pth_file.stat().st_mtime
                    ).isoformat(),
                    "metadata": metadata,
                })
        serializer = ModelInfoSerializer(models, many=True)
        return Response(serializer.data)


class ModelRegistryView(AsyncAPIView):
    """Full model artifact registry with disk sync."""

    async def get(self, request):
        db = await ensure_db_connection()
        svc = ModelRegistryService(db)
        sync = request.query_params.get("sync", "true").lower() != "false"
        model_type = request.query_params.get("model_type")
        vertical = request.query_params.get("vertical")
        data = await svc.sync_and_list(
            sync=sync,
            model_type=model_type,
            vertical=vertical,
        )
        artifacts = ModelArtifactSerializer(data["artifacts"], many=True)
        return Response({
            "sync": data["sync"],
            "total": data["total"],
            "artifacts": artifacts.data,
            "assignments": data["assignments"],
        })


class ModelRegistryTableView(AsyncAPIView):
    """Flattened rows for UI data tables."""

    async def get(self, request):
        db = await ensure_db_connection()
        rows = await ModelRegistryService(db).get_registry_table()
        serializer = ModelRegistryTableRowSerializer(rows, many=True)
        return Response({"rows": serializer.data, "total": len(rows)})


class ModelArtifactDetailView(AsyncAPIView):
    async def get(self, request, artifact_id):
        db = await ensure_db_connection()
        from backend.api.repositories.model_repository import ModelArtifactRepository
        repo = ModelArtifactRepository(db, MODEL_DIR)
        doc = await repo.get_artifact(artifact_id)
        if not doc:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(ModelArtifactSerializer(doc).data)


class ModelSyncView(AsyncAPIView):
    async def post(self, request):
        db = await ensure_db_connection()
        stats = await ModelRegistryService(db).repo.sync_from_disk()
        return Response({"status": "ok", **stats})


class ModelUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        upload = request.FILES.get("file")
        if not upload:
            return Response(
                {"detail": "No file"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not upload.name.endswith(".pth"):
            return Response(
                {"detail": "Only .pth files"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODEL_DIR / upload.name
        with open(dest, "wb") as out:
            for chunk in upload.chunks():
                out.write(chunk)
        return Response(
            {"status": "uploaded", "filename": upload.name, "size_mb": dest.stat().st_size / 1e6},
            status=status.HTTP_201_CREATED,
        )
