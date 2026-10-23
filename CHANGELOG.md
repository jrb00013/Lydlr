# Changelog

All notable changes to Lydlr are documented here.

## [Unreleased]

### Added

- Vision codec (`temporal_dct_rans`): temporal residual, 8×8 DCT, countable rANS bits. Image-only `lydlr` benches use it and beat intra JPEG at matched bpp (`scripts/prove_lydlr_beats_jpeg.py`).
- Console **Beat JPEG** page ships the measured proof (`frontend/public/beat-jpeg-proof.json`). CI fails if that file drifts from a fresh run.
- Compression waterfall panel on Dashboard and Visual Monitoring.
- Torch-free `bitrate_math` and `clip_stats` utilities with unit tests.
- LYDT wire protocol and link-policy pytest coverage (ROS-free).
- Ops runbook, CI-local guide, and HTTP API surface map.
- `scripts/health_probe.py`, `scripts/check_fixtures.py`, `scripts/summarize_rd_json.py`.
- `CONTRIBUTING.md`, `SECURITY.md`, Makefile `ci-smoke` target, expanded CI test matrix.

### Changed

- Communication package lazily imports `rclpy` QoS helpers so unit tests run without ROS 2.
