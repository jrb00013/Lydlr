"""Orchestrates model artifact sync and node assignments."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.api.repositories.model_repository import ModelArtifactRepository
from backend.api.repositories.node_repository import NodeRepository


class ModelRegistryService:
    def __init__(self, db):
        model_dir = Path(os.getenv("MODEL_DIR", "/app/models"))
        self.repo = ModelArtifactRepository(db, model_dir)
        self.nodes = NodeRepository(db)

    async def sync_and_list(
        self,
        *,
        sync: bool = True,
        model_type: Optional[str] = None,
        vertical: Optional[str] = None,
    ) -> Dict[str, Any]:
        sync_stats = await self.repo.sync_from_disk() if sync else {"scanned": 0, "upserted": 0}
        artifacts = await self.repo.list_artifacts(
            model_type=model_type,
            vertical=vertical,
        )
        assignments = await self.repo.list_assignments()
        return {
            "sync": sync_stats,
            "artifacts": artifacts,
            "assignments": assignments,
            "total": len(artifacts),
        }

    async def get_registry_table(self) -> List[Dict[str, Any]]:
        await self.repo.sync_from_disk()
        artifacts = await self.repo.list_artifacts()
        assignments = {a["node_id"]: a for a in await self.repo.list_assignments()}
        rows = []
        for art in artifacts:
            deployed_nodes = [
                nid for nid, a in assignments.items()
                if a.get("artifact_id") == art["artifact_id"]
                or a.get("model_version") == art["version"]
            ]
            perf = art.get("performance") or {}
            training = art.get("training") or {}
            rows.append({
                "artifact_id": art["artifact_id"],
                "version": art["version"],
                "model_type": art["model_type"],
                "architecture": art["architecture"],
                "status": art["status"],
                "size_mb": art.get("size_mb", 0),
                "vertical_targets": art.get("vertical_targets", []),
                "modalities": art.get("modalities", []),
                "compression_ratio": perf.get("compression_ratio"),
                "quality_score": perf.get("quality_score"),
                "inference_ms": perf.get("inference_time_ms"),
                "training_source": training.get("data_source"),
                "epochs": training.get("epochs"),
                "deployed_node_count": len(deployed_nodes),
                "deployed_nodes": deployed_nodes,
                "filename": art.get("filename"),
                "updated_at": art.get("updated_at"),
            })
        return rows
