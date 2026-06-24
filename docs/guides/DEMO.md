# Lydlr 5-Minute Demo

## What you'll see

End-to-end edge compression with an RL-tuned link budget loop: sensor ingest → ML compressor → uplink → fleet health dashboard — all running locally.

## Prerequisites

```bash
docker compose up -d            # MongoDB, Redis, backend API
cd frontend && npm start        # → http://localhost:3000
```

## Step 1 — Generate replay fixtures

```bash
python scripts/generate_replay_fixtures.py --seed 42
# → scripts/fixture_{drone,iot,warehouse}_clip.npz
```

## Step 2 — Benchmark link policy

```bash
python scripts/bench_link_policy.py
# CSV output: 180 rows across 9 scenarios (drone/iot/highcap × nominal/burst/low_quality/latency)
```

## Step 3 — Start edge compressor (ROS2)

```bash
LYDLR_SENSOR_SOURCE=replay \
LYDLR_VERTICAL=drone \
LYDLR_REPLAY_PATH=scripts/fixture_drone_clip.npz \
ros2 run lydlr_ai edge_compressor_node
```

The node loads the fixture, runs multimodal compression, and reports metrics via LYDT wire protocol.

## Step 4 — Open the dashboard

http://localhost:3000 shows:
- Fleet panel: drone + IoT nodes with compression/latency stats
- **Link Budget Health**: per-node kbps vs budget bar chart with quality indicators
- **Live telemetry**: compression ratio, latency, quality time series
- Uplink efficiency chart

## Step 5 — Toggle RL policy

Toggle `RL mode` in the RL Policy panel to switch between heuristic `target_compression_level()` and the trained PPO controller. The panel shows:
- Current controller mode (heuristic / PPO)
- RL action and reward
- Running reward trend

## Key metrics

| Metric | What it measures |
|--------|-----------------|
| Compression ratio | × reduction in payload size |
| Latency | ms per compression cycle |
| Quality score | LPIPS-based predicted perceptual quality (0–1) |
| Budget utilization | % of uplink budget consumed |

## What makes it work

- `link_policy.py`: uplink budget rules for drone (512 kbps) vs IoT (64 kbps)
- `edge_compressor_node.py`: LPIPS quality guard + modality gating + RL controller
- `wire.py`: LYDT binary protocol for metrics/coordination/compressed frames
- `distributed_coordinator.py`: fleet-level bandwidth orchestration
- `rl_policy.py`: PPO compression controller trained on link-budget simulation
