# ROS2 Communication & Transport Architecture

## Layer stack

```
Sensors (shared bus)     Edge compressor        Fleet control           Ground link
────────────────────     ───────────────        ─────────────           ────────────
/camera/image_raw   →    LYDT compress    →    coordinator        →    transport_relay
/lidar/data              /lydlr/{id}/          deployment_mgr          /lydlr/ground/
/imu/data                  transport/*         communication_hub         uplink/*
/audio/data
/cmd_vel
```

## Wire protocol (LYDT v1)

Binary frames on `std_msgs/UInt8MultiArray` — see `lydlr_ai/communication/wire.py`.

| msg_type | Purpose |
|----------|---------|
| 1 COMPRESSED | zlib(tensor) + metadata |
| 2 METRICS | compression, latency, quality, bytes in/out |
| 3 COORDINATION | fleet target compression + Mbps allocation |
| 4 HEARTBEAT | node alive + model version |

## Topic graph

| Topic | QoS | Publisher | Subscriber |
|-------|-----|-----------|------------|
| `/lydlr/{id}/transport/compressed` | best-effort | edge | relay, coordinator |
| `/lydlr/{id}/transport/metrics` | best-effort | edge | hub, coordinator, deploy mgr |
| `/lydlr/{id}/coordination` | reliable | coordinator | edge |
| `/lydlr/{id}/command/deploy` | reliable transient | deploy mgr | edge |
| `/lydlr/fleet/deploy` | reliable | API / CLI | all edges |
| `/lydlr/fleet/status` | reliable | comm hub | loggers |
| `/lydlr/ground/uplink/compressed` | best-effort | relay | GCS mock |

Legacy topics `/{id}/metrics` and `/{id}/compressed` remain for backward compatibility.

## QoS policies

Defined in `lydlr_ai/communication/qos.py`:

- **sensor_ingress** — BEST_EFFORT, depth 5 (drop frames under CPU load)
- **compressed_egress** — BEST_EFFORT (low latency over reliability)
- **command** — RELIABLE + TRANSIENT_LOCAL (deploy survives late join)
- **coordination** — RELIABLE
- **metrics** — BEST_EFFORT, depth 20

## Nodes

| Executable | Role |
|------------|------|
| `edge_compressor_node` | Compress + publish LYDT + HTTP metrics |
| `distributed_coordinator` | Bandwidth allocation + coordination signals |
| `model_deployment_manager` | Hot-swap models, rollback |
| `transport_relay` | Shape ground uplink to `GROUND_UPLINK_MBPS` |
| `communication_hub` | Fleet status JSON on `/lydlr/fleet/status` |
| `synthetic_multimodal_publisher` | Drone/IoT sensor simulation |

## Launch

```bash
# Full transport stack
ros2 launch lydlr_ai drone_iot_transport.launch.py

# Or shell launcher
./start-lydlr.sh --build -d --ros2
```

## Deploy a model

```bash
# Per node
ros2 topic pub --once /lydlr/node_0/command/deploy std_msgs/String "data: 'vv1.0'"

# Entire fleet
ros2 topic pub --once /lydlr/fleet/deploy std_msgs/String "data: 'all:vv1.0'"
```

## Inspect transport

```bash
ros2 topic list | grep lydlr
ros2 topic echo /lydlr/fleet/status
ros2 topic hz /lydlr/node_0/transport/compressed
```
