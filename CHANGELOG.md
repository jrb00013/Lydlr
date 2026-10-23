# Changelog

All notable changes to Lydlr are documented here.

## [Unreleased]

### Added

- Compression waterfall panel on Dashboard and Visual Monitoring.
- Torch-free `bitrate_math` and `clip_stats` utilities with unit tests.
- LYDT wire protocol and link-policy pytest coverage (ROS-free).
- Ops runbook, CI-local guide, and HTTP API surface map.
- `scripts/health_probe.py`, `scripts/check_fixtures.py`, `scripts/summarize_rd_json.py`.
- `CONTRIBUTING.md`, `SECURITY.md`, Makefile `ci-smoke` target, expanded CI test matrix.

### Changed

- Communication package lazily imports `rclpy` QoS helpers so unit tests run without ROS 2.
