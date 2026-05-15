"""Deployment audit repository."""
from typing import Any, Dict, List, Optional

from backend.api.schema.collections import COLLECTIONS
from backend.api.schema.documents import build_deployment_document


class DeploymentRepository:
    def __init__(self, db):
        self.collection = db[COLLECTIONS["DEPLOYMENTS"]]

    async def create(self, model_version: str, node_ids: List[str], **kwargs) -> Dict[str, Any]:
        doc = build_deployment_document(model_version, node_ids, **kwargs)
        await self.collection.insert_one(doc)
        doc.pop("_id", None)
        return doc

    async def update_status(
        self,
        deployment_id: str,
        status: str,
        results: Optional[List[Dict]] = None,
    ) -> None:
        from datetime import datetime, timezone
        update: Dict[str, Any] = {
            "status": status,
            "completed_at": datetime.now(timezone.utc),
        }
        if results is not None:
            update["results"] = results
        await self.collection.update_one(
            {"deployment_id": deployment_id},
            {"$set": update},
        )

    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = await self.collection.find({}).sort("deployed_at", -1).limit(limit).to_list(limit)
        for r in rows:
            r.pop("_id", None)
        return rows
