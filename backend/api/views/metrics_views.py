"""
Metrics and statistics views — raw samples, rollups, fleet tables.
"""
import csv
import io
import json
import logging
from datetime import datetime

from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.serializers import (
    CompressionMetricsSerializer,
    SystemStatsSerializer,
    PaginatedMetricsSerializer,
    MetricsRollupRowSerializer,
)
from backend.api.connections import redis_client
from backend.api.redis_pubsub import publish_message
from backend.api.services.metrics_service import MetricsService
from backend.api.repositories.node_repository import NodeRepository

logger = logging.getLogger(__name__)


class MetricsView(AsyncAPIView):
    """Get and store metrics"""

    async def get(self, request):
        fmt = request.query_params.get("format", "json")
        node_id = request.query_params.get("node_id")
        vertical = request.query_params.get("vertical")
        limit = int(request.query_params.get("limit", 100))
        skip = int(request.query_params.get("skip", 0))

        db = await ensure_db_connection()
        svc = MetricsService(db)

        if fmt == "table":
            data = await svc.query_table(
                node_id=node_id,
                vertical=vertical,
                limit=limit,
                skip=skip,
            )
            return Response(PaginatedMetricsSerializer(data).data)

        metrics = await svc.metrics.list_samples(
            node_id=node_id,
            vertical=vertical,
            limit=limit,
            skip=skip,
        )
        for m in metrics:
            m.pop("_id", None)
        serializer = CompressionMetricsSerializer(metrics, many=True)
        return Response(serializer.data)

    async def post(self, request):
        db = await ensure_db_connection()
        serializer = CompressionMetricsSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        doc = serializer.validated_data
        svc = MetricsService(db)
        await svc.ingest(doc)

        await redis_client.setex(
            f"metrics:{doc['node_id']}:latest",
            60,
            json.dumps(doc, default=str),
        )
        await publish_message("metrics_update", doc)

        return Response({"status": "success"})


class MetricsRollupsView(AsyncAPIView):
    async def get(self, request):
        db = await ensure_db_connection()
        node_id = request.query_params.get("node_id")
        limit = int(request.query_params.get("limit", 72))
        rows = await MetricsService(db).rollups_table(node_id=node_id, limit=limit)
        return Response({
            "rows": MetricsRollupRowSerializer(rows, many=True).data,
            "total": len(rows),
        })


class MetricsFleetView(AsyncAPIView):
    async def get(self, request):
        db = await ensure_db_connection()
        summary = await MetricsService(db).fleet_summary()
        return Response(summary)


class SystemStatsView(AsyncAPIView):
    """Get overall system statistics"""

    async def get(self, request):
        db = await ensure_db_connection()
        nodes = NodeRepository(db)
        from backend.api.services.metrics_service import MetricsService

        total_nodes = await nodes.count()
        active_nodes = await nodes.count({"status": "active"})
        active_drones = await nodes.count({"vertical": "drone", "status": "active"})
        active_iot = await nodes.count({"vertical": "iot", "status": "active"})

        fleet_cfg = await db.system_config.find_one({"type": "fleet_profile"})
        node_cfg = await db.system_config.find_one({"type": "node_configuration"})

        agg = await MetricsService(db).metrics.fleet_aggregates()
        recent_metrics = await MetricsService(db).metrics.list_samples(limit=100)

        avg_compression = agg.get("avg_compression_ratio", 0)
        avg_latency = agg.get("avg_latency_ms", 0)
        avg_quality = agg.get("avg_quality_score", 0)

        fleet_nodes = await nodes.list_all()
        node_budget = {n["node_id"]: n.get("uplink_budget_kbps", 256) for n in fleet_nodes}
        saved_kbps = 0.0
        if recent_metrics and avg_compression > 1:
            by_node = {}
            for m in recent_metrics:
                nid = m.get("node_id")
                if nid and nid not in by_node:
                    by_node[nid] = m
            for nid, m in by_node.items():
                budget = node_budget.get(nid, 256)
                ratio = m.get("compression_ratio", 1.0)
                if ratio > 1:
                    saved_kbps += budget * (1.0 - 1.0 / ratio)

        stats = {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "active_drones": active_drones,
            "active_iot": active_iot,
            "avg_compression_ratio": avg_compression,
            "avg_latency_ms": avg_latency,
            "avg_quality_score": avg_quality,
            "fleet_profile": (fleet_cfg or {}).get("name", "drone_iot_edge"),
            "vertical": (node_cfg or {}).get("vertical", "drone"),
            "estimated_uplink_saved_kbps": round(saved_kbps, 1),
            "metrics_sample_count": agg.get("sample_count", 0),
            "by_vertical": agg.get("by_vertical", {}),
            "timestamp": datetime.utcnow(),
        }

        serializer = SystemStatsSerializer(stats)
        return Response(serializer.data)


class MetricsExportView(AsyncAPIView):
    """Export metrics samples as JSON or CSV for Grafana / offline analysis."""

    async def get(self, request):
        db = await ensure_db_connection()
        export_fmt = request.query_params.get("format", "json").lower()
        node_id = request.query_params.get("node_id")
        vertical = request.query_params.get("vertical")
        limit = int(request.query_params.get("limit", 1000))

        metrics = await MetricsService(db).metrics.list_samples(
            node_id=node_id,
            vertical=vertical,
            limit=limit,
        )
        for m in metrics:
            m.pop("_id", None)
            if isinstance(m.get("timestamp"), datetime):
                m["timestamp"] = m["timestamp"].isoformat()

        if export_fmt == "csv":
            buffer = io.StringIO()
            fields = [
                "timestamp", "node_id", "vertical", "compression_ratio",
                "latency_ms", "quality_score", "compression_level",
                "bandwidth_estimate", "bytes_in", "bytes_out",
                "modality_bytes_in", "modality_bytes_out", "modality_quality",
            ]
            writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)
            response = HttpResponse(buffer.getvalue(), content_type="text/csv")
            response["Content-Disposition"] = 'attachment; filename="lydlr_metrics.csv"'
            return response

        return Response({"rows": metrics, "total": len(metrics)})
