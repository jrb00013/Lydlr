.PHONY: help test ci-smoke fixtures health prove-beat bench-smoke

PYTHON ?= python3
export PYTHONPATH := ros2/src/lydlr_ai:$(PYTHONPATH)

help:
	@echo "Targets: test ci-smoke fixtures health prove-beat bench-smoke"

test:
	$(PYTHON) -m pytest tests/ -q

ci-smoke: test prove-beat bench-smoke
	@echo "ci-smoke ok"

fixtures:
	$(PYTHON) scripts/check_fixtures.py

health:
	$(PYTHON) scripts/health_probe.py

prove-beat:
	$(PYTHON) scripts/prove_beat_jpeg.py \
		--fixture scripts/fixture_correlated_clip.npz --frames 12 --max-width 160
	$(PYTHON) scripts/prove_lydlr_beats_jpeg.py \
		--fixture scripts/fixture_correlated_clip.npz --frames 8 --max-width 96

bench-smoke:
	mkdir -p scripts/results
	$(PYTHON) scripts/bench_codecs.py \
		--clip scripts/fixture_correlated_clip.npz --codecs jpeg,lydlr --target-bpps 0.5 \
		--max-frames 6 \
		--out scripts/results/ci_rd_smoke.json
