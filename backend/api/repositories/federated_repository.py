"""Federated learning round repository."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.api.schema.collections import COLLECTIONS
from backend.api.schema.documents import build_federated_round_document


class FederatedRoundRepository:
    def __init__(self, db):
        self.collection = db[COLLECTIONS["FEDERATED_ROUNDS"]]

    async def create(
        self,
        participant_node_ids: List[str],
        *,
        base_version: str,
        max_delta_kbps: float = 128.0,
        inference_backend: str = "onnx",
    ) -> Dict[str, Any]:
        doc = build_federated_round_document(
            participant_node_ids,
            base_version=base_version,
            max_delta_kbps=max_delta_kbps,
            inference_backend=inference_backend,
        )
        await self.collection.insert_one(doc)
        doc.pop("_id", None)
        return doc

    async def get(self, round_id: str) -> Optional[Dict[str, Any]]:
        row = await self.collection.find_one({"round_id": round_id})
        if not row:
            return None
        row.pop("_id", None)
        return row

    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = await self.collection.find({}).sort("created_at", -1).limit(limit).to_list(limit)
        for row in rows:
            row.pop("_id", None)
        return rows

    async def update(self, round_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self.collection.update_one({"round_id": round_id}, {"$set": fields})
        return await self.get(round_id)
