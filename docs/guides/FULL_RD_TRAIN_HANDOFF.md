# Full Rate–Distortion Train — Handoff

**Last updated:** 2026-08-03  
**Goal:** Train `EnhancedMultimodalCompressor` v2 under `L = D + λR` with multi-frame temporal coding.  
**Hardware:** any CUDA GPU with enough free VRAM for batch 1 at 480×640 (use `--cpu` only if needed).

Related design: [NEURAL_COMPRESSION_RD_PLAN.md](../architecture/NEURAL_COMPRESSION_RD_PLAN.md)

---

## What “full real train” means

| Tier | Preset | Epochs × steps × seq | Wall-clock (rough) | Purpose |
|------|--------|----------------------|--------------------|---------|
| Smoke | `--preset smoke` | 1 × 2 × 1 | minutes | Sanity only |
| Short | `--preset short` | 25 × 40 × 2 | tens of minutes on GPU | Quick curve check |
| **Full synthetic RD** | `--preset full` | **100 × 100 × 4** | **a few hours on a mid-range GPU** | Real deal on synthetic multimodal |
| Full + real sensor data | custom + collected dataset | 20–50 epochs over ≥1k sequences | overnight typical | Production-quality weights |

**Full synthetic RD (this handoff’s target):** `--preset full`  
Total optimizer updates ≈ `100 × 100 × 4 = 40,000` forward/backward passes at 480×640 with ResNet18 VAE backbone.

Exact ETA depends on your GPU. The trainer prints measured `s/step`, `elapsed`, and `eta` every epoch — trust those over this table.

---

## Prerequisites

1. **GPU + drivers**
   ```bash
   nvidia-smi
   python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
   ```
2. **Python deps:** `torch`, `torchvision`, `lpips` (optional for LPIPS eval), repo on `PYTHONPATH`
   ```bash
   cd /path/to/Lydlr
   export PYTHONPATH=ros2/src/lydlr_ai
   ```
3. **Disk:** `models/` and `training_logs/` writable (both gitignored for `.pth`)
4. **VRAM:** start with `--batch-size 1`. On OOM, keep batch 1 and free other GPU users first; only then shrink resolution in code.
5. **Do not commit** large `.pth` files (already in `.gitignore`).

---

## Start the full train (now)

```bash
cd /home/joeblack/Documents/Lydlr
mkdir -p models training_logs
export PYTHONPATH=ros2/src/lydlr_ai

python3 scripts/train_rd_compressor.py --preset full \
  --batch-size 1 --lr 1e-4 --save-every 10 \
  --out-dir models \
  2>&1 | tee training_logs/rd_full_$(date -u +%Y%m%d_%H%M%S).log
```

Artifacts written under `models/`:

| File | Role |
|------|------|
| `lydlr_compressor_v2_full_latest.pth` | **Resume pointer** (always overwritten) |
| `lydlr_compressor_v2_full_<stamp>_eN.pth` | Periodic epoch checkpoints |
| `lydlr_compressor_v2_full_<stamp>_final.pth` | Finished weights |
| `FULL_TRAIN_STATUS.json` | Last epoch / paths / metrics tail |
| `metadata_lydlr_compressor_v2_*.json` | Per-checkpoint metadata |

---

## Where to pick up later

### 1. Check if a run is alive

```bash
pgrep -af train_rd_compressor || true
tail -n 40 training_logs/rd_full_*.log | tail -40
cat models/FULL_TRAIN_STATUS.json 2>/dev/null
```

### 2. Resume after stop / sleep / crash

```bash
export PYTHONPATH=ros2/src/lydlr_ai
python3 scripts/train_rd_compressor.py --preset full \
  --resume models/lydlr_compressor_v2_full_latest.pth \
  --batch-size 1 --save-every 10 \
  2>&1 | tee -a training_logs/rd_full_resume_$(date -u +%Y%m%d_%H%M%S).log
```

Resume restores model (+ optimizer when compatible) and continues from `epoch + 1`.

### 3. Eval the best/latest checkpoint

```bash
export PYTHONPATH=ros2/src/lydlr_ai
python3 scripts/eval_compression_rd.py --frames 32 \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth
python3 scripts/eval_compression_rd.py --frames 16 --edge-fast \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth
```

Look for: **R (rate_bits) trending down** as λ anneals up, **PSNR/SSIM up**, keyframe fraction ≈ `1/keyframe_period`.

### 4. Deploy to edge (when happy)

Copy `lydlr_compressor_v2_full_*_final.pth` (+ matching metadata) into the node model dir expected by `ModelRegistry` (`models/<node_id>/`). Loads use `strict=False` for new RD modules.

---

## Leave-off checklist (before you walk away)

- [ ] Full train command running (or intentionally stopped with latest checkpoint saved)
- [ ] Note the log file under `training_logs/`
- [ ] Confirm `models/lydlr_compressor_v2_full_latest.pth` exists after first `--save-every` hit
- [ ] Optional: `nvidia-smi` shows python using GPU memory
- [ ] Next session: resume from latest, then eval, then only then commit *code* changes (never weights)

---

## After full synthetic — next upgrade (real data)

Structured synthetic (band-limited scenes + shared ego-motion) is the default in
`scripts/train_rd_compressor.py` / `eval_compression_rd.py`. See
[TRAINING_DATA_APPLIED_MATH.md](../architecture/TRAINING_DATA_APPLIED_MATH.md).

Noise-trained checkpoints are **not** comparable on PSNR — restart `--preset full`
after pulling the structured-data change (do not resume noise weights for quality eval).

Not required to finish the synthetic full train, but needed for “production”:

1. Collect sequences: `ros2 launch lydlr_ai collect_training_data.launch.py`
2. Point a future data loader at `data/training_data/` (still mostly synthetic in v2 script today)
3. Re-run `--preset full` with real batches swapped in
4. Sweep `λ ∈ {0.01, 0.05, 0.1}` and pick the RD operating point for the link budget

Estimated overnight job with ≥1000 real sequences on a typical mid-range GPU.

---

## Known gaps (do not block this train)

- Entropy rate is a differentiable proxy, not an ANS/arithmetic bitstream yet
- Callers in `enhanced_train.py` / `test_enhanced_system.py` may still unpack the old 8-tuple — prefer `unpack_compressor_output` / this script
- LPIPS not required for the RD loop; add later for perceptual reporting

---

## Success criteria for this full train

1. Completes 100 epochs (or resume reaches 100) without OOM  
2. `FULL_TRAIN_STATUS.json` shows finite D/R/L  
3. Eval PSNR clearly above the untrained ~10 dB baseline  
4. Mean `rate_bits` below the untrained ~512 ceiling after λ anneal  

When those hold, treat weights as the v2 candidate for edge hot-swap.
