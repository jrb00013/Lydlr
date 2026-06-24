"""Federated learning rounds API."""
from __future__ import annotations

import csv
import io
import logging

from django.http import HttpResponse
from rest_framework.response import Response

from backend.api.repositories.federated_repository import FederatedRoundRepository
from backend.api.views.base import AsyncAPIView, ensure_db_connection

logger = logging.getLogger(__name__)


def _rounds_to_csv(rounds: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "round_id",
        "status",
        "base_version",
        "merged_version",
        "participants",
        "max_delta_kbps",
        "modality_bytes_out_total",
        "inference_backend",
        "created_at",
        "completed_at",
    ])
    for r in rounds:
        writer.writerow([
            r.get("round_id", ""),
            r.get("status", ""),
            r.get("base_version", ""),
            r.get("merged_version", ""),
            ",".join(r.get("participant_node_ids", [])),
            r.get("max_delta_kbps", ""),
            r.get("modality_bytes_out_total", 0),
            r.get("inference_backend", ""),
            r.get("created_at", ""),
            r.get("completed_at", ""),
        ])
    return buf.getvalue()


class FederatedRoundListView(AsyncAPIView):
    """GET list rounds; POST start a round; ?format=csv for export."""

    async def get(self, request):
        db = await ensure_db_connection()
        repo = FederatedRoundRepository(db)
        rounds = await repo.list_recent()
        if request.query_params.get("format") == "csv":
            csv_text = _rounds_to_csv(rounds)
            resp = HttpResponse(csv_text, content_type="text/csv")
            resp["Content-Disposition"] = 'attachment; filename="federated_rounds.csv"'
            return resp
        return Response(rounds)

    async def post(self, request):
        body = request.data or {}
        participant_node_ids = body.get("participant_node_ids") or []
        base_version = body.get("base_version")
        if not participant_node_ids:
            return Response({"error": "participant_node_ids required"}, status=400)
        if not base_version:
            return Response({"error": "base_version required"}, status=400)

        db = await ensure_db_connection()
        nodes = db.nodes
        missing = []
        for nid in participant_node_ids:
            found = await nodes.find_one({"node_id": nid})
            if not found:
                missing.append(nid)
        if missing:
            return Response({"error": "unknown nodes", "missing": missing}, status=400)

        repo = FederatedRoundRepository(db)
        doc = await repo.create(
            participant_node_ids,
            base_version=str(base_version),
            max_delta_kbps=float(body.get("max_delta_kbps", 128)),
            inference_backend=str(body.get("inference_backend", "onnx")),
        )
        logger.info("Started federated round %s for %s", doc["round_id"], participant_node_ids)
        return Response(doc, status=201)


class FederatedRoundDetailView(AsyncAPIView):
    async def get(self, request, round_id: str):
        db = await ensure_db_connection()
        repo = FederatedRoundRepository(db)
        doc = await repo.get(round_id)
        if not doc:
            return Response({"error": "round not found"}, status=404)
        return Response(doc)
