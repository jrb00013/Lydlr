# Contributing to Lydlr

Thanks for helping improve the adaptive edge compression stack.

## Ground rules

1. **Countable bits only** for rate claims — see `docs/architecture/BENCHMARK_PROTOCOL.md`.
2. Beat-JPEG proofs must use temporally correlated clips (`scripts/fixture_correlated_clip.npz`), not independent multimodal fixtures.
3. Do not commit checkpoints, ONNX/TensorRT bundles, or `.env` secrets (see `.gitignore`).
4. Prefer small, reviewable PRs with tests for math/wire helpers.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# Frontend
cd frontend && npm install && npm start
```

## Tests

```bash
PYTHONPATH=ros2/src/lydlr_ai pytest tests/ -q
make ci-smoke
```

## PR checklist

- [ ] `pytest tests/` green locally
- [ ] Beat-JPEG / bench smoke if you touched codecs or fixtures
- [ ] Docs updated when changing protocol or API surface
- [ ] No large binaries added
