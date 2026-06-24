"""Metrics ingestion and query orchestration."""
from typing import Any, Dict, List, Optional

from backend.api.repositories.metrics_repository import MetricsRepository
from backend.api.repositories.node_repository import NodeRepository


class MetricsService:
    def __init__(self, db):
        self.metrics = MetricsRepository(db)
        self.nodes = NodeRepository(db)

    async def ingest(self, doc: Dict[str, Any]) -> None:
        node = await self.nodes.get(doc["node_id"])
        if node and not doc.get("vertical"):
            doc["vertical"] = node.get("vertical")
        await self.metrics.insert_sample(doc)

    async def query_table(
        self,
        *,
        node_id: Optional[str] = None,
        vertical: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> Dict[str, Any]:
        rows = await self.metrics.list_samples(
            node_id=node_id,
            vertical=vertical,
            limit=limit,
            skip=skip,
        )
        total = await self.metrics.count_samples(
            {"node_id": node_id} if node_id else (
                {"vertical": vertical} if vertical else {}
            )
        )
        table = []
        for m in rows:
            table.append({
                "node_id": m.get("node_id"),
                "vertical": m.get("vertical", "—"),
                "timestamp": m.get("timestamp"),
                "compression_ratio": round(m.get("compression_ratio", 0), 3),
                "latency_ms": round(m.get("latency_ms", 0), 2),
                "quality_score": round(m.get("quality_score", 0), 4),
                "compression_level": round(m.get("compression_level", 0), 3),
                "bandwidth_estimate": round(m.get("bandwidth_estimate", 0), 3),
                "bytes_in": m.get("bytes_in"),
                "bytes_out": m.get("bytes_out"),
                "modality_bytes_in": m.get("modality_bytes_in"),
                "modality_bytes_out": m.get("modality_bytes_out"),
                "modality_quality": m.get("modality_quality"),
            })
        return {"rows": table, "total": total, "limit": limit, "skip": skip}

    async def rollups_table(
        self,
        *,
        node_id: Optional[str] = None,
        limit: int = 72,
    ) -> List[Dict[str, Any]]:
        rollups = await self.metrics.list_rollups(node_id=node_id, limit=limit)
        return [
            {
                "rollup_key": r.get("rollup_key"),
                "node_id": r.get("node_id"),
                "vertical": r.get("vertical"),
                "bucket_start": r.get("bucket_start"),
                "samples": r.get("sample_count", 0),
                "avg_compression": round(r.get("avg_compression_ratio", 0), 3),
                "avg_latency_ms": round(r.get("avg_latency_ms", 0), 2),
                "avg_quality": round(r.get("avg_quality_score", 0), 4),
                "min_compression": r.get("min_compression_ratio"),
                "max_compression": r.get("max_compression_ratio"),
            }
            for r in rollups
        ]

    async def fleet_summary(self) -> Dict[str, Any]:
        agg = await self.metrics.fleet_aggregates()
        latest = await self.metrics.latest_per_node()
        return {"aggregates": agg, "latest_by_node": latest}
