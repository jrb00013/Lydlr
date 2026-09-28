# Benchmark protocol (matched-rate RD)

This document defines how Lydlr compares against JPEG and other baselines. The harness lives in `lydlr_ai.model.bench_baselines` and is invoked from `scripts/bench_codecs.py` and `scripts/prove_beat_jpeg.py`.

## Countable bits only

A reported **rate** is always **countable payload**: encoded file bytes for image/video baselines, or rANS (or fixed-length index) bits for the Lydlr path. The differentiable **entropy proxy** from training may be logged separately; it is **never** promoted to `bits` or `bpp` in comparison tables.

If no real entropy coder is available, the harness reports `rate_source=fixed_length_bits` and sets `bits` to the fixed-length index budget exactly—still countable, still honest.

## Matched-rate definition

For each target **bits-per-pixel (bpp)** on a fixed clip:

1. Baseline codecs search a quality parameter until encoded bpp is within a **relative tolerance** (default 10%) of the target, or flag `matched=false` with a reason.
2. Distortion (PSNR, SSIM; optional LPIPS) is measured **after decode** at that operating point—not at a different rate.
3. Comparisons are **matched-rate**: two codecs are compared at the same target bpp, not at arbitrary quality knobs.

Interpolation along a measured RD curve is allowed only **inside** the measured bpp range (`interp_rd`); extrapolation is forbidden for claims.

## No proxy-as-bits

Marketing-style “effective bandwidth” or training-time cross-entropy must not stand in for wire rate. Claims in README and papers must trace to harness output where `rate_source` and `bits` are explicit.

## Primary “beat JPEG” axis: temporal residual

JPEG (and independent WebP/H.264-intra frames) have **no temporal memory**: each frame is coded alone. Sensor video on robots and drones is **strongly correlated in time**.

Lydlr’s falsifiable claim is on **temporal residual coding**—predicting from recent frames and spending bits on what changed—beating **intra JPEG at matched clip bpp** on **real recorded clips**, not on band-limited synthetic Markov scenes alone. Synthetic data is useful for regression tests; **beat-JPEG conclusions require real NPZ clips** (`scripts/record_sensor_clip.py`) and a trained checkpoint.

Run the published check:

```bash
PYTHONPATH=ros2/src/lydlr_ai python scripts/prove_beat_jpeg.py
PYTHONPATH=ros2/src/lydlr_ai python scripts/bench_codecs.py --clip path/to/real.npz --codecs jpeg,lydlr
```

Numbers in documentation must come from these scripts, not from placeholders.
