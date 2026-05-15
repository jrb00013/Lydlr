"""
Metrics persistence + hourly rollups.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.api.schema.collections import COLLECTIONS
from backend.api.schema.documents import build_metrics_rollup_key


class MetricsRepository:
    def __init__(self, db):
        self.db = db
        self.metrics = db[COLLECTIONS["METRICS"]]
        self.rollups = db[COLLECTIONS["METRICS_ROLLUPS"]]

    async def insert_sample(self, doc: Dict[str, Any]) -> None:
        await self.metrics.insert_one(doc)
        await self._update_rollup(doc)

    async def _update_rollup(self, doc: Dict[str, Any]) -> None:
        ts = doc.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts is None:
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        bucket_start = ts.replace(minute=0, second=0, microsecond=0)
        node_id = doc["node_id"]
        rollup_key = build_metrics_rollup_key(node_id, bucket_start)

        inc_fields = {
            "sample_count": 1,
            "sum_compression_ratio": doc.get("compression_ratio", 0),
            "sum_latency_ms": doc.get("latency_ms", 0),
            "sum_quality_score": doc.get("quality_score", 0),
        }
        set_on_insert = {
            "rollup_key": rollup_key,
            "node_id": node_id,
            "vertical": doc.get("vertical"),
            "bucket_start": bucket_start,
            "min_compression_ratio": doc.get("compression_ratio", 0),
            "max_compression_ratio": doc.get("compression_ratio", 0),
            "min_latency_ms": doc.get("latency_ms", 0),
            "max_latency_ms": doc.get("latency_ms", 0),
        }

        existing = await self.rollups.find_one({"rollup_key": rollup_key})
        if existing:
            min_cr = min(existing.get("min_compression_ratio", inc_fields["sum_compression_ratio"]),
                         doc.get("compression_ratio", 0))
            max_cr = max(existing.get("max_compression_ratio", 0), doc.get("compression_ratio", 0))
            min_lat = min(existing.get("min_latency_ms", doc.get("latency_ms", 0)), doc.get("latency_ms", 0))
            max_lat = max(existing.get("max_latency_ms", 0), doc.get("latency_ms", 0))
            await self.rollups.update_one(
                {"rollup_key": rollup_key},
                {
                    "$inc": inc_fields,
                    "$set": {
                        "min_compression_ratio": min_cr,
                        "max_compression_ratio": max_cr,
                        "min_latency_ms": min_lat,
                        "max_latency_ms": max_lat,
                        "last_sample_at": ts,
                    },
                },
            )
        else:
            await self.rollups.insert_one({
                **set_on_insert,
                **inc_fields,
                "last_sample_at": ts,
            })

    async def list_samples(
        self,
        *,
        node_id: Optional[str] = None,
        vertical: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if node_id:
            query["node_id"] = node_id
        if vertical:
            query["vertical"] = vertical
        cursor = self.metrics.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        return await cursor.to_list(limit)

    async def count_samples(self, query: Optional[Dict] = None) -> int:
        return await self.metrics.count_documents(query or {})

    async def list_rollups(
        self,
        *,
        node_id: Optional[str] = None,
        limit: int = 168,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if node_id:
            query["node_id"] = node_id
        cursor = self.rollups.find(query).sort("bucket_start", -1).limit(limit)
        rows = await cursor.to_list(limit)
        for r in rows:
            n = r.get("sample_count") or 1
            r["avg_compression_ratio"] = r.get("sum_compression_ratio", 0) / n
            r["avg_latency_ms"] = r.get("sum_latency_ms", 0) / n
            r["avg_quality_score"] = r.get("sum_quality_score", 0) / n
        return rows

    async def latest_per_node(self) -> Dict[str, Dict[str, Any]]:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": "$node_id", "doc": {"$first": "$$ROOT"}}},
        ]
        result = {}
        async for row in self.metrics.aggregate(pipeline):
            doc = row["doc"]
            result[doc["node_id"]] = doc
        return result

    async def fleet_aggregates(self, sample_limit: int = 200) -> Dict[str, Any]:
        recent = await self.list_samples(limit=sample_limit)
        if not recent:
            return {
                "sample_count": 0,
                "avg_compression_ratio": 0,
                "avg_latency_ms": 0,
                "avg_quality_score": 0,
                "by_vertical": {},
                "by_node": {},
            }
        by_vertical: Dict[str, List[float]] = {}
        by_node: Dict[str, List[float]] = {}
        for m in recent:
            v = m.get("vertical") or "unknown"
            by_vertical.setdefault(v, []).append(m.get("compression_ratio", 0))
            by_node.setdefault(m["node_id"], []).append(m.get("compression_ratio", 0))

        def _avg(vals):
            return sum(vals) / len(vals) if vals else 0

        return {
            "sample_count": len(recent),
            "avg_compression_ratio": _avg([m.get("compression_ratio", 0) for m in recent]),
            "avg_latency_ms": _avg([m.get("latency_ms", 0) for m in recent]),
            "avg_quality_score": _avg([m.get("quality_score", 0) for m in recent]),
            "by_vertical": {k: {"avg_compression": _avg(v)} for k, v in by_vertical.items()},
            "by_node": {k: {"avg_compression": _avg(v)} for k, v in by_node.items()},
        }
