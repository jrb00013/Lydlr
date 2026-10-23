# Lydlr HTTP API surface (console)

High-level map of the Django/DRF backend used by the edge console. Paths are relative to the API base (`REACT_APP_API_URL`, default `http://localhost:8000`).

## Fleet & nodes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/nodes/` | List edge compressor nodes |
| GET | `/api/nodes/{id}/` | Node detail + last metrics |
| GET | `/api/fleet/stats/` | Aggregate drones / IoT / compression |

## Metrics & previews

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/metrics/` | Recent compression metrics |
| WS | `/ws/metrics/` | Live metric push (when enabled) |
| GET | `/api/preview/{node}/{side}.jpg` | JPEG preview (`raw`, `reconstructed`, `heatmap`) |
| GET | `/api/preview/{node}/mjpeg` | MJPEG stream |

## Models & deploy

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/models/` | Model registry |
| POST | `/api/deploy/` | Schedule edge deploy |
| GET | `/api/federated/` | Federated sync status |

## Health

Prefer `GET /api/health/` (or the project’s existing health route) from `scripts/health_probe.py --api`.

Exact route names may vary slightly by backend app version — treat this as the console contract, and confirm against `backend/` URLConf when integrating new clients.
