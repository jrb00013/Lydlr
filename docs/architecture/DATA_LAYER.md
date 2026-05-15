# Lydlr Data Layer Architecture

## Overview

Lydlr splits **control plane** (MongoDB + API + React tables) from **data plane** (ROS2 + PyTorch on disk).

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE (MongoDB)                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ nodes        │ model_       │ metrics      │ node_model_        │
│ devices      │ artifacts    │ metrics_     │ assignments        │
│ sensors      │              │ rollups      │ deployments        │
│ system_config│              │ (TTL 7d)     │                    │
└──────┬───────┴──────┬───────┴──────┬───────┴─────────┬──────────┘
       │              │              │                 │
       ▼              ▼              ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI (repositories + services)                   │
│  /api/models/registry/table/  /api/metrics/?format=table       │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PLANE                                    │
│  ros2: edge_compressor_node → POST /api/metrics/               │
│  disk:  *.pth + metadata_*.json → sync → model_artifacts         │
└─────────────────────────────────────────────────────────────────┘
```

## Collections

### `nodes`
Edge compressor fleet members.

| Field | Type | Description |
|-------|------|-------------|
| node_id | string | Primary key |
| vertical | drone \| iot | Product vertical |
| display_name | string | UI label |
| status | active \| stopped | Fleet status |
| model_version | string | Currently assigned version |
| uplink_budget_kbps | number | Link budget for savings calc |
| config | object | compression_level, target_quality |
| sensors | string[] | Subscribed modalities |

### `model_artifacts`
Registry row per `.pth` file (synced from disk).

| Field | Type | Description |
|-------|------|-------------|
| artifact_id | string | `{model_type}_{version}` |
| version | string | e.g. `vv1.0` |
| model_type | string | multimodal_compressor, sensor_motor_compressor |
| architecture | string | EnhancedMultimodalCompressor |
| status | registered \| production \| deprecated |
| vertical_targets | string[] | drone, iot |
| training | object | data_source, epochs, optimizer |
| performance | object | compression_ratio, quality_score, inference_time_ms |
| checksum_sha256 | string | File integrity |

### `node_model_assignments`
One document per node — which artifact is active.

### `metrics` (TTL 7 days)
Raw telemetry from edge compressors.

| Field | Type |
|-------|------|
| node_id, vertical, timestamp | |
| compression_ratio, latency_ms, quality_score | |
| compression_level, bandwidth_estimate | |
| bytes_in, bytes_out | optional |

### `metrics_rollups`
Hourly aggregates per node (`rollup_key = node_id:YYYYMMDDHH`).

### `deployments`
Audit log of model pushes to nodes.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/models/registry/table/` | Sortable UI table rows |
| POST | `/api/models/sync/` | Scan disk → upsert artifacts |
| GET | `/api/metrics/?format=table` | Paginated sample rows |
| GET | `/api/metrics/rollups/` | Hourly rollup table |
| GET | `/api/metrics/fleet/` | Aggregates + latest per node |
| POST | `/api/metrics/` | Edge ingest (ROS2 reporter) |

## Code Layout

```
backend/api/
  schema/          # Document builders, index specs
  repositories/    # MongoDB access
  services/        # Orchestration (sync, rollups, tables)
  views/           # HTTP handlers
```

## ML Models (data plane)

| Artifact | Class | Role |
|----------|-------|------|
| `lydlr_compressor_v*.pth` | EnhancedMultimodalCompressor | Camera, LiDAR, IMU, audio |
| `*_sensor_motor_v*.pth` | SensorMotorCompressor | Actuator + proprioception |
| `metadata_*.json` | — | Training + performance metadata |

Train: `ros2 run lydlr_ai train_synthetic_models`  
Sync: `POST /api/models/sync/` or automatic on API startup.

## Frontend Tables

- **Models** → `GET /api/models/registry/table/`
- **Metrics** → raw samples + hourly rollups tabs

Shared component: `frontend/src/components/ui/DataTable.js`
