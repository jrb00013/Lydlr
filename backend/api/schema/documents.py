"""
Document builders — single place for field names and defaults.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib


def _utcnow():
    return datetime.now(timezone.utc)


def build_node_document(
    node_id: str,
    *,
    status: str = "active",
    vertical: str = "drone",
    display_name: Optional[str] = None,
    node_type: str = "edge_compressor",
    model_version: Optional[str] = None,
    uplink_budget_kbps: float = 512,
    sensors: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = _utcnow()
    return {
        "node_id": node_id,
        "status": status,
        "vertical": vertical,
        "display_name": display_name or node_id,
        "node_type": node_type,
        "model_version": model_version,
        "uplink_budget_kbps": uplink_budget_kbps,
        "sensors": sensors or ["camera", "lidar", "imu", "audio"],
        "config": config or {
            "compression_level": 0.8,
            "target_quality": 0.85,
            "enable_metrics": True,
        },
        "created_at": now,
        "last_update": now,
    }


def build_device_document(
    device_id: str,
    device_type: str,
    *,
    node_id: Optional[str] = None,
    vertical: str = "drone",
    status: str = "online",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    now = _utcnow()
    return {
        "device_id": device_id,
        "device_type": device_type,
        "node_id": node_id,
        "vertical": vertical,
        "status": status,
        "description": description or "",
        "created_at": now,
        "last_update": now,
        "sensors": [],
        "metadata": {},
    }


def build_metric_document(
    node_id: str,
    compression_ratio: float,
    latency_ms: float,
    quality_score: float,
    *,
    bandwidth_estimate: float = 1.0,
    compression_level: float = 0.8,
    vertical: Optional[str] = None,
    bytes_in: Optional[int] = None,
    bytes_out: Optional[int] = None,
) -> Dict[str, Any]:
    doc = {
        "node_id": node_id,
        "compression_ratio": float(compression_ratio),
        "latency_ms": float(latency_ms),
        "quality_score": float(quality_score),
        "bandwidth_estimate": float(bandwidth_estimate),
        "compression_level": float(compression_level),
        "timestamp": _utcnow(),
    }
    if vertical:
        doc["vertical"] = vertical
    if bytes_in is not None:
        doc["bytes_in"] = int(bytes_in)
    if bytes_out is not None:
        doc["bytes_out"] = int(bytes_out)
    return doc


def build_metrics_rollup_key(node_id: str, bucket_start: datetime) -> str:
    return f"{node_id}:{bucket_start.strftime('%Y%m%d%H')}"


def build_model_artifact_document(
    version: str,
    filename: str,
    *,
    model_type: str = "multimodal_compressor",
    architecture: str = "EnhancedMultimodalCompressor",
    size_bytes: int = 0,
    file_path: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    vertical_targets: Optional[List[str]] = None,
    status: str = "registered",
) -> Dict[str, Any]:
    now = _utcnow()
    artifact_id = f"{model_type}_{version}"
    meta = metadata or {}
    return {
        "artifact_id": artifact_id,
        "version": version,
        "model_type": model_type,
        "architecture": architecture,
        "filename": filename,
        "file_path": file_path,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "checksum_sha256": meta.get("checksum_sha256", ""),
        "vertical_targets": vertical_targets or ["drone", "iot"],
        "status": status,
        "training": {
            "data_source": meta.get("training_data_source", "synthetic_multimodal"),
            "epochs": meta.get("training_epochs"),
            "batch_size": meta.get("batch_size"),
            "optimizer": meta.get("optimizer", "adam"),
            "use_synthetic": meta.get("use_synthetic_data", True),
            "completed_at": meta.get("training_date") or meta.get("timestamp"),
        },
        "performance": {
            "compression_ratio": meta.get("final_compression_ratio"),
            "quality_score": meta.get("final_quality"),
            "psnr": meta.get("psnr"),
            "ssim": meta.get("ssim"),
            "inference_time_ms": meta.get("inference_time_ms"),
        },
        "modalities": meta.get(
            "modalities",
            ["camera", "lidar", "imu", "audio"],
        ),
        "parameters": meta.get("parameters"),
        "device_target": meta.get("device", "cuda|cpu"),
        "metadata_raw": meta,
        "created_at": now,
        "updated_at": now,
    }


def build_deployment_document(
    model_version: str,
    node_ids: List[str],
    *,
    deployment_id: Optional[str] = None,
    status: str = "deploying",
    initiated_by: str = "api",
) -> Dict[str, Any]:
    now = _utcnow()
    dep_id = deployment_id or hashlib.sha256(
        f"{model_version}:{','.join(sorted(node_ids))}:{now.isoformat()}".encode()
    ).hexdigest()[:16]
    return {
        "deployment_id": dep_id,
        "model_version": model_version,
        "node_ids": node_ids,
        "status": status,
        "initiated_by": initiated_by,
        "deployed_at": now,
        "completed_at": None,
        "results": [],
    }


def build_federated_round_document(
    participant_node_ids: List[str],
    *,
    base_version: str,
    max_delta_kbps: float = 128.0,
    inference_backend: str = "onnx",
    round_id: Optional[str] = None,
    status: str = "pending",
    initiated_by: str = "coordinator",
) -> Dict[str, Any]:
    now = _utcnow()
    rid = round_id or hashlib.sha256(
        f"{base_version}:{','.join(sorted(participant_node_ids))}:{now.isoformat()}".encode()
    ).hexdigest()[:16]
    full_id = rid if str(rid).startswith("fr_") else f"fr_{rid}"
    participants = [
        {
            "node_id": nid,
            "status": "pending",
            "delta_sha256": "",
            "modality_bytes_out": 0,
        }
        for nid in participant_node_ids
    ]
    participant_status = {
        nid: {
            "status": "pending",
            "delta_kbps": 0,
            "checksum_sha256": None,
            "modality_bytes_out": 0,
        }
        for nid in participant_node_ids
    }
    return {
        "round_id": full_id,
        "status": status,
        "base_version": base_version,
        "merged_version": None,
        "participant_node_ids": list(participant_node_ids),
        "participants": participants,
        "participant_status": participant_status,
        "max_delta_kbps": float(max_delta_kbps),
        "modality_bytes_out_total": 0,
        "inference_backend": inference_backend,
        "initiated_by": initiated_by,
        "deltas": [],
        "created_at": now,
        "started_at": now,
        "completed_at": None,
    }


def build_node_assignment(node_id: str, model_version: str, artifact_id: str) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "model_version": model_version,
        "artifact_id": artifact_id,
        "status": "active",
        "assigned_at": _utcnow(),
    }

