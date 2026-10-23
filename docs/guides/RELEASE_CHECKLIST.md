# Release checklist

Before tagging or promoting a deploy bundle:

1. `make ci-smoke` is green on a clean checkout.
2. Beat-JPEG claim still holds on `fixture_correlated_clip.npz`.
3. `scripts/validate_deploy_bundle.py` passes for the candidate bundle.
4. CHANGELOG `[Unreleased]` section moved under a version heading.
5. Frontend production build: `cd frontend && npm run build`.
6. Spot-check console Dashboard waterfall + Visual Monitoring previews against a live or demo pulse fleet.
7. Confirm `.env` secrets are not in the release artifact.
