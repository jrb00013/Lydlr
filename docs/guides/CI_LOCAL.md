# Running CI locally

Mirror `.github/workflows/ci.yml` before opening a PR.

```bash
python3 -m pip install --upgrade pip
pip install numpy pillow opencv-python-headless pytest
# optional (true_rate / torch paths):
pip install torch --index-url https://download.pytorch.org/whl/cpu

PYTHONPATH=ros2/src/lydlr_ai python3 -c "from lydlr_ai.model import bench_baselines; print('ok')"
PYTHONPATH=ros2/src/lydlr_ai pytest tests/ -q

PYTHONPATH=ros2/src/lydlr_ai python3 scripts/prove_beat_jpeg.py \
  --fixture scripts/fixture_correlated_clip.npz --frames 12 --max-width 160

PYTHONPATH=ros2/src/lydlr_ai python3 scripts/bench_codecs.py \
  --clip scripts/fixture_correlated_clip.npz --codecs jpeg,webp --target-bpps 0.2 \
  --out scripts/results/ci_rd_smoke.json
```

Or: `make ci-smoke` if Make is available.
