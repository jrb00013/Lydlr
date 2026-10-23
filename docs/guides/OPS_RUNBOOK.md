# Lydlr ops runbook

Day-2 operations for the edge console + ROS 2 compression fleet.

## Bring-up

```bash
./start-lydlr.sh --build -d --ros2
# after colcon build:
ros2 launch lydlr_ai drone_iot_transport.launch.py
```

## Health checks

```bash
PYTHONPATH=ros2/src/lydlr_ai python3 scripts/health_probe.py
PYTHONPATH=ros2/src/lydlr_ai python3 scripts/health_probe.py --api http://localhost:8000
PYTHONPATH=ros2/src/lydlr_ai python3 scripts/check_fixtures.py
```

## Beat-JPEG / RD smoke

```bash
PYTHONPATH=ros2/src/lydlr_ai python3 scripts/prove_beat_jpeg.py \
  --fixture scripts/fixture_correlated_clip.npz --frames 12 --max-width 160
PYTHONPATH=ros2/src/lydlr_ai python3 scripts/bench_codecs.py \
  --clip scripts/fixture_correlated_clip.npz --codecs jpeg,webp --target-bpps 0.2 \
  --out scripts/results/local_rd_smoke.json
PYTHONPATH=ros2/src/lydlr_ai python3 scripts/summarize_rd_json.py scripts/results/local_rd_smoke.json
```

## Common failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Console empty / no metrics | Backend or ROS nodes down | `docker compose ps`, restart with `--ros2` |
| Beat-JPEG proof fails on drone fixture | Fixture is multimodal samples, not video | Use `fixture_correlated_clip.npz` |
| `rclpy` import errors in unit tests | Host has no ROS 2 | Import `wire` / `link_policy` directly; CI is ROS-free |
| Camera wedge on Orin | nvargus race | Prefer V4L2; see `scripts/orin_safe_probe.py` |

## Rollback

Pin deploy bundles via `scripts/validate_deploy_bundle.py` before promoting a new ONNX/TensorRT artifact. Keep the previous bundle directory until the fleet reports healthy heartbeats.
