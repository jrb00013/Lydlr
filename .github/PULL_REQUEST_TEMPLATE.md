## Summary

<!-- What changed and why -->

## Test plan

- [ ] `PYTHONPATH=ros2/src/lydlr_ai pytest tests/ -q`
- [ ] `make ci-smoke` (or prove_beat_jpeg + bench_codecs smoke)
- [ ] Frontend smoke if UI touched (`cd frontend && npm start`)

## Checklist

- [ ] Rate claims use countable bits / documented fixtures
- [ ] No secrets, checkpoints, or deploy bundles committed
- [ ] Docs updated when protocol or ops steps change
