"""
Model artifact registry — DB is source of truth for UI; disk is source of weights.
"""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.api.schema.collections import COLLECTIONS
from backend.api.schema.documents import build_model_artifact_document


class ModelArtifactRepository:
    def __init__(self, db, model_dir: Path):
        self.db = db
        self.collection = db[COLLECTIONS["MODEL_ARTIFACTS"]]
        self.assignments = db[COLLECTIONS["NODE_MODEL_ASSIGNMENTS"]]
        self.model_dir = model_dir

    def _parse_version_from_stem(self, stem: str) -> str:
        if "_v" in stem:
            return stem.split("_v", 1)[1]
        return stem

    def _find_metadata(self, pth_path: Path) -> Dict[str, Any]:
        stem = pth_path.stem
        version = self._parse_version_from_stem(stem)
        candidates = [
            self.model_dir / f"metadata_{stem}.json",
            self.model_dir / f"metadata_{version}.json",
            self.model_dir / f"metadata_lydlr_compressor_v{version}.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        return {}

    def _infer_model_type(self, filename: str) -> str:
        lower = filename.lower()
        if "sensor_motor" in lower:
            return "sensor_motor_compressor"
        return "multimodal_compressor"

    def _file_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def scan_disk_artifacts(self) -> List[Dict[str, Any]]:
        artifacts = []
        if not self.model_dir.exists():
            return artifacts
        for pth in sorted(self.model_dir.glob("*.pth")):
            meta = self._find_metadata(pth)
            version = meta.get("version") or self._parse_version_from_stem(pth.stem)
            meta["checksum_sha256"] = self._file_checksum(pth)
            doc = build_model_artifact_document(
                version=version,
                filename=pth.name,
                model_type=self._infer_model_type(pth.name),
                architecture=meta.get("architecture", "EnhancedMultimodalCompressor"),
                size_bytes=pth.stat().st_size,
                file_path=str(pth),
                metadata=meta,
                status="production" if version.endswith("1.0") or "v1" in version else "registered",
            )
            artifacts.append(doc)
        return artifacts

    async def sync_from_disk(self) -> Dict[str, int]:
        scanned = self.scan_disk_artifacts()
        upserted = 0
        for doc in scanned:
            doc["updated_at"] = datetime.now(timezone.utc)
            result = await self.collection.update_one(
                {"artifact_id": doc["artifact_id"]},
                {"$set": doc, "$setOnInsert": {"created_at": doc["created_at"]}},
                upsert=True,
            )
            if result.upserted_id or result.modified_count:
                upserted += 1
        return {"scanned": len(scanned), "upserted": upserted}

    async def list_artifacts(
        self,
        *,
        model_type: Optional[str] = None,
        vertical: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if model_type:
            query["model_type"] = model_type
        if status:
            query["status"] = status
        if vertical:
            query["vertical_targets"] = vertical
        cursor = (
            self.collection.find(query)
            .sort("updated_at", -1)
            .skip(skip)
            .limit(limit)
        )
        rows = await cursor.to_list(limit)
        for r in rows:
            r.pop("_id", None)
        return rows

    async def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        doc = await self.collection.find_one({"artifact_id": artifact_id})
        if doc:
            doc.pop("_id", None)
        return doc

    async def get_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        doc = await self.collection.find_one({"version": version})
        if doc:
            doc.pop("_id", None)
        return doc

    async def assign_to_node(self, node_id: str, version: str, artifact_id: str) -> None:
        from backend.api.schema.documents import build_node_assignment
        doc = build_node_assignment(node_id, version, artifact_id)
        await self.assignments.update_one(
            {"node_id": node_id},
            {"$set": doc},
            upsert=True,
        )

    async def list_assignments(self) -> List[Dict[str, Any]]:
        rows = await self.assignments.find({}).to_list(200)
        for r in rows:
            r.pop("_id", None)
        return rows

    async def count(self, query: Optional[Dict] = None) -> int:
        return await self.collection.count_documents(query or {})
