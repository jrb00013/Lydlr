# Full Rate–Distortion Train — Handoff

**Last updated:** 2026-08-03  
**Goal:** Train `EnhancedMultimodalCompressor` v2 under `L = D + λR` with multi-frame temporal coding.  
**Hardware:** any CUDA GPU with enough free VRAM for batch 1 at 480×640 (use `--cpu` only if needed).

Related design: [NEURAL_COMPRESSION_RD_PLAN.md](../architecture/NEURAL_COMPRESSION_RD_PLAN.md) · data math: [TRAINING_DATA_APPLIED_MATH.md](../architecture/TRAINING_DATA_APPLIED_MATH.md) · stability: [RD_STABILITY_APPLIED_MATH.md](../architecture/RD_STABILITY_APPLIED_MATH.md)

---

## Completed run (structured synthetic, tempered RD)

| Item | Value |
|------|--------|
| Preset | `--preset full` (100 × 100 × 4) resumed after KL tempering |
| Wall clock | ~46 min on CUDA for epochs 10→100 (full job ~1 h class) |
| Final weights | `models/lydlr_compressor_v2_full_20260803_211009_final.pth` (gitignored) |
| Resume pointer | `models/lydlr_compressor_v2_full_latest.pth` |
| Train R | ~443 → **0.16** bits (proxy) |
| Train Drec | **~0.018 flat** (continue-gate passed: R↓ without recon collapse) |
| Eval PSNR / SSIM / MSE | **17.6 dB / 0.18 / 0.017** on structured clips |
| Eval vs noise baseline | was ~9 dB / MSE 0.15 on noise-trained weights |
| p50 latency | ~19 ms |

**Decision rule used during the run:** continue while R falls and Drec stays flat; stop + applied-math fix if R↓ with Drec/PSNR blow-up or KL uncapped spikes. KL tempering (`RD_STABILITY_APPLIED_MATH.md`) prevented the earlier D~6000 failure mode.

**Watch next:** entropy `R → ~0` with flat Drec means the *learned rate proxy* collapsed (overconfident symbol model), not necessarily a true near-zero bitstream. Next math pass should separate **proxy R** from **countable index bits** / ANS before claiming link bitrate.

---

## What “full real train” means

| Tier | Preset | Epochs × steps × seq | Wall-clock (rough) | Purpose |
|------|--------|----------------------|--------------------|---------|
| Smoke | `--preset smoke` | 1 × 2 × 1 | minutes | Sanity only |
| Short | `--preset short` | 25 × 40 × 2 | tens of minutes on GPU | Quick curve check |
| **Full synthetic RD** | `--preset full` | **100 × 100 × 4** | **a few hours on a mid-range GPU** (this box: ~1 h) | Real deal on synthetic multimodal |
| Full + real sensor data | custom + collected dataset | 20–50 epochs over ≥1k sequences | overnight typical | Production-quality weights |

Exact ETA depends on your GPU. The trainer prints measured `s/step`, `elapsed`, and `eta` every epoch.

---

## Prerequisites

1. **GPU + drivers**
   ```bash
   nvidia-smi
   python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
   ```
2. **Python deps:** `torch`, `torchvision`; optional `lpips`
   ```bash
   export PYTHONPATH=ros2/src/lydlr_ai
   ```
3. **Disk:** `models/` and `training_logs/` writable (`.pth` gitignored)
4. **VRAM:** start with `--batch-size 1`

---

## Start / resume

```bash
mkdir -p models training_logs
export PYTHONPATH=ros2/src/lydlr_ai

PYTHONUNBUFFERED=1 python3 scripts/train_rd_compressor.py --preset full \
  --batch-size 1 --lr 1e-4 --save-every 10 --cut-prob 0.03 \
  --out-dir models \
  2>&1 | tee training_logs/rd_full_$(date -u +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 python3 scripts/train_rd_compressor.py --preset full \
  --resume models/lydlr_compressor_v2_full_latest.pth \
  --batch-size 1 --save-every 10 \
  2>&1 | tee -a training_logs/rd_full_resume_$(date -u +%Y%m%d_%H%M%S).log
```

| File | Role |
|------|------|
| `lydlr_compressor_v2_full_latest.pth` | Resume pointer |
| `lydlr_compressor_v2_full_*_final.pth` | Finished weights |
| `FULL_TRAIN_STATUS.json` | Last status |

---

## Eval

```bash
export PYTHONPATH=ros2/src/lydlr_ai
python3 scripts/eval_compression_rd.py --frames 32 \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth \
  --out models/eval_structured_final.json
```

---

## Next session

1. **True rate accounting** — countable quantized indices / ANS vs entropy proxy.  
2. Collect real sequences and retrain.  
3. Deploy final `.pth` into `models/<node_id>/` for edge hot-swap.

---

## Leave-off checklist

- [x] Full structured train completed to epoch 100  
- [x] Final + latest checkpoints on disk (gitignored)  
- [x] Final eval (~17.6 dB PSNR)  
- [ ] Real-data collection + true-rate math  
- [ ] Edge hot-swap smoke  

Pickup: eval/deploy above; do not resume the finished 100-epoch run unless extending epochs.
