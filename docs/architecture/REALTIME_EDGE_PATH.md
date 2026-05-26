# Real-time edge compression path

## What Lydlr does

Data moves **camera / LiDAR / IMU → NVIDIA edge (Orin/Jetson) → compressed uplink → control plane**.

```
Camera (sensor_msgs/Image)
        │
        ▼
  /camera/image_raw  ──►  edge_compressor_node  ──►  LYDT compressed frame
        │                        │                      │
        │                        ├── metrics ──────────►│ HTTP POST /api/metrics/
        │                        │                      │ WebSocket → Dashboard
        │                        └── coordination ◄────┤ /api/fleet/link-policy/
        ▼
distributed_coordinator (link budget → compression level)
```

Compression runs **on the edge before transfer**, so you send fewer bits without blindly destroying quality. The coordinator reads each node’s **uplink budget (kbps)** and live **bytes_out** and raises/lowers compression level to stay under budget while respecting **min quality**.

## Per-device specs (Mongo → ROS)

Set on each fleet node (UI: **Nodes → Link spec** or API):

| Field | UAV example | IoT example |
|-------|-------------|-------------|
| `uplink_budget_kbps` | 512 | 64 |
| `min_quality` | 0.75 | 0.65 |
| `vision_fps_cap` | 15 | 2 |
| `max_latency_ms` | 50 | 80 |

```bash
curl -X PATCH http://localhost:8000/api/nodes/node_0/link-spec/ \
  -H 'Content-Type: application/json' \
  -d '{"uplink_budget_kbps":512,"min_quality":0.8,"vision_fps_cap":15}'
```

Coordinator polls `GET /api/fleet/link-policy/` every 10s (or `FLEET_LINK_POLICY_JSON` env).

## Sensor sources

| Mode | Env | Use |
|------|-----|-----|
| Synthetic | `LYDLR_SENSOR_SOURCE=synthetic` | Dev / CI |
| Replay clip | `LYDLR_SENSOR_SOURCE=replay` | Recorded NPZ (`scripts/record_sensor_clip.py`) |
| USB camera | `LYDLR_SENSOR_SOURCE=camera` | Real V4L2 → `/camera/image_raw` |
| Rosbag | `LYDLR_SENSOR_SOURCE=rosbag` | `LYDLR_ROSBAG_PATH=/path/to/bag` |

```bash
# Record demo clip
python3 scripts/record_sensor_clip.py --vertical drone

# Launch with replay ingest (not synthetic)
LYDLR_SENSOR_SOURCE=replay ./start-lydlr.sh -d --ros2
```

## Jetson deploy bundle

```bash
python3 scripts/export_onnx_bundle.py --version vv1.0 --out deploy_bundles/
```

Produces ONNX + `manifest.json` for TensorRT on Orin. Deploy via **Deploy** page copies weights and publishes ROS deploy topic.

## Metrics you care about

- **compression_ratio** — bytes in / bytes out  
- **latency_ms** — edge inference + pack time  
- **quality_score** — perceptual guard (LPIPS loop)  
- **estimated_throughput_kbps** — live uplink use vs **uplink_budget_kbps**

Export history: `GET /api/metrics/export/?format=csv`
