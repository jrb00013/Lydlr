# Changelog

All notable changes to Lydlr are documented here.

## [Unreleased]

### Added

- Vision codec v3 (`temporal_dct_rans_v3`): YCbCr 4:2:0 chroma + mild frequency-weighted DCT. Further rate drop vs v2; console shows **model confidence** next to compression.
- Vision codec v2 (`temporal_dct_rans_v2`): YCbCr temporal residual, mild frequency-weighted 8×8 DCT, countable rANS. Emits `compression_ratio`, `zero_frac`, and `confidence` (predictor lock + PSNR). Console live session defaults to qstep=8.
- Vision codec (`temporal_dct_rans`): temporal residual, 8×8 DCT, countable rANS bits. Image-only `lydlr` benches use it and beat intra JPEG at matched bpp (`scripts/prove_lydlr_beats_jpeg.py`).
- Console **Beat JPEG** page ships the measured proof (`frontend/public/beat-jpeg-proof.json`). CI fails if that file drifts from a fresh run.
- `GET /api/codec/frame/` encodes the next correlated-clip frame with the vision codec. The dashboard waterfall and previews use those countable bytes.
- Host API defaults to localhost Mongo/Redis with short connect timeouts so a missing database does not hang the console.
- Compression waterfall panel on Dashboard and Visual Monitoring.
- Torch-free `bitrate_math` and `clip_stats` utilities with unit tests.
- LYDT wire protocol and link-policy pytest coverage (ROS-free).
- Ops runbook, CI-local guide, and HTTP API surface map.
- `scripts/health_probe.py`, `scripts/check_fixtures.py`, `scripts/summarize_rd_json.py`.
- `CONTRIBUTING.md`, `SECURITY.md`, Makefile `ci-smoke` target, expanded CI test matrix.

### Changed

- Beat-JPEG proof accepts **below_floor** wins (rate under JPEG’s floor at higher PSNR) and drift-checks `best_margin_db`.
- Communication package lazily imports `rclpy` QoS helpers so unit tests run without ROS 2.
