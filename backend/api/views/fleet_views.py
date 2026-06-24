"""Fleet link policy API for ROS coordinator."""
import logging

from rest_framework.response import Response

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.services.link_policy_service import build_fleet_link_policy

logger = logging.getLogger(__name__)


class FleetLinkPolicyView(AsyncAPIView):
    async def get(self, request):
        db = await ensure_db_connection()
        return Response(await build_fleet_link_policy(db))


class FleetLinkHealthView(AsyncAPIView):
    """
    Per-node link budget health status.

    Compares estimated throughput (from latest metrics) against
    uplink_budget_kbps for each node. Returns over/under/at-budget
    classification for fleet monitoring.
    """

    async def get(self, request):
        db = await ensure_db_connection()
        fleet = await build_fleet_link_policy(db)
        nodes = fleet.get("nodes", {})

        rows = []
        over_count = 0
        under_count = 0
        at_budget_count = 0

        for node_id, spec in nodes.items():
            budget = spec.get("uplink_budget_kbps", 512)
            latest = await db.metrics.find_one(
                {"node_id": node_id},
                sort=[("timestamp", -1)],
            )
            if latest:
                bytes_out = latest.get("bytes_out") or 0
                throughput = (bytes_out * 8) / 0.1 / 1000.0 if bytes_out else 0.0
                quality = latest.get("quality_score", 0)
                latency = latest.get("latency_ms", 0)
                ratio_in = latest.get("compression_ratio", 0)
            else:
                throughput = 0.0
                quality = 0.0
                latency = 0.0
                ratio_in = 0.0

            utilization = min(throughput / max(budget, 1), 2.0)
            if utilization > 0.95:
                status = "over_budget"
                over_count += 1
            elif utilization < 0.5:
                status = "under_budget"
                under_count += 1
            else:
                status = "at_budget"
                at_budget_count += 1

            vertical = spec.get("vertical", "drone")
            min_quality = spec.get("min_quality", 0.7)
            quality_ok = quality >= min_quality

            rows.append({
                "node_id": node_id,
                "vertical": vertical,
                "uplink_budget_kbps": budget,
                "estimated_throughput_kbps": round(throughput, 1),
                "budget_utilization": round(utilization, 3),
                "status": status,
                "quality_score": round(quality, 4),
                "quality_ok": quality_ok,
                "latency_ms": round(latency, 1),
                "compression_ratio": round(ratio_in, 2),
                "min_quality_threshold": min_quality,
            })

        return Response({
            "nodes": rows,
            "summary": {
                "total": len(rows),
                "over_budget": over_count,
                "under_budget": under_count,
                "at_budget": at_budget_count,
                "nodes_with_quality_issues": sum(
                    1 for r in rows if not r["quality_ok"]
                ),
            },
        })
