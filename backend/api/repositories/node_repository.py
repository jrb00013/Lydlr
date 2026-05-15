"""
Edge node fleet repository.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.api.schema.collections import COLLECTIONS
from backend.api.schema.documents import build_node_document


class NodeRepository:
    def __init__(self, db):
        self.collection = db[COLLECTIONS["NODES"]]

    async def list_all(self, *, vertical: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {"vertical": vertical} if vertical else {}
        rows = await self.collection.find(query).sort("node_id", 1).to_list(200)
        for r in rows:
            r.pop("_id", None)
        return rows

    async def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        doc = await self.collection.find_one({"node_id": node_id})
        if doc:
            doc.pop("_id", None)
        return doc

    async def upsert(self, doc: Dict[str, Any]) -> None:
        node_id = doc["node_id"]
        doc["last_update"] = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"node_id": node_id},
            {"$set": doc, "$setOnInsert": {"created_at": doc.get("created_at", doc["last_update"])}},
            upsert=True,
        )

    async def create(self, node_id: str, **kwargs) -> Dict[str, Any]:
        doc = build_node_document(node_id, **kwargs)
        await self.upsert(doc)
        return doc

    async def update_fields(self, node_id: str, fields: Dict[str, Any]) -> bool:
        fields["last_update"] = datetime.now(timezone.utc)
        result = await self.collection.update_one(
            {"node_id": node_id},
            {"$set": fields},
        )
        return result.matched_count > 0

    async def delete(self, node_id: str) -> bool:
        result = await self.collection.delete_one({"node_id": node_id})
        return result.deleted_count > 0

    async def count(self, query: Optional[Dict] = None) -> int:
        return await self.collection.count_documents(query or {})
