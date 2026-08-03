# Training Data — Applied-Math Discovery Notes

**Status:** active  
**Problem observed:** RD train metrics improved (R↓, train D↓) while eval PSNR stayed ~9 dB.  
**Root cause (failed guess recorded):** treating i.i.d. `U(0,1)` pixels as “images.”

---

## System (plain language)

A moving platform looks at a world that mostly stays the same from one moment to the next. Bright patches, ground, and obstacles drift slowly across the camera. Motion sensors roughly agree with how the picture slides. Laser points belong to the same solid surfaces. Sound changes more slowly than the picture. The compressor’s job is to send the *change* plus occasional full snapshots — so the data must contain real, measurable change structure, not fresh static every frame.

---

## Inventory

| Kind | Items |
|------|--------|
| Entities | Scene field (slow), moving blobs (objects), camera ego-motion, LiDAR returns, IMU rates, audio envelope |
| Actions | Translate/rotate viewpoint; objects drift; rare scene cuts |
| Measurables | Pixel intensity [0,1]; optical flow (px/frame); residual energy ‖x_t−x_{t−1}‖²; LiDAR range (m); IMU (m/s², rad/s); audio energy |
| Constraints | Intensities ∈ [0,1]; motion continuous except cuts; modalities must share the same ego-motion when coupled |

---

## Representations examined

1. **Diagram:** slow background field + K blobs + ego translation `(vx, vy)` driving all sensors.  
2. **Time series:** residual energy high on cuts, low on smooth motion — white noise stays flat-high every frame (no temporal structure).  
3. **Hand example:** 8×8 patch, blob shift 1 px → residual sparse; replace with new noise → residual ≈ 2× variance, no compressible pattern.

---

## Candidate invariants

| Candidate | Result |
|-----------|--------|
| Pixel histogram roughly stable under small ego-motion | Holds for structured scenes; **broken** by i.i.d. redraw |
| Residual energy ≪ signal energy when motion is slow | Holds iff frames share a latent scene state |
| Cross-modal: IMU ∝ ego acceleration / camera flow | Soft invariant we enforce in the generator |
| Rate R → 0 for static scene after first keyframe | Must be approximately true for a correct codec+data pair |

**Break attempt on noise data:** static “scene” redrawn each step → residual never shrinks → temporal modules cannot learn. Data, not only the model, violated the invariant.

---

## Symmetries (what they rule out)

- **Time-shift:** dynamics depend on relative motion → generator must be a **stateful** process `S_t = F(S_{t−1}, u_t)`, not independent draws.  
- **Translation (approx.):** shifting blobs should not change statistics → use translation-equivariant scene construction.  
- **Description:** intensity scale is arbitrary in [0,1] → prefer structure (edges, motion) over absolute gray level.  

Rules out: `torch.rand` per frame; independent LiDAR/IMU/audio with no shared `u_t`.

---

## Dimensionless groups

| Group | Meaning |
|-------|---------|
| `φ = ‖x_t − x_{t−1}‖ / ‖x_t‖` | Relative residual (target: small on smooth motion) |
| `ν = v · Δt / L` | Motion in scene-widths per frame (keep `ν ≪ 1` usually) |
| `κ = 1/K` | Keyframe density in the train loop (must match codec) |
| `ρ_modal = R_image / R_total` | Not forced yet; keep modalities coupled so ρ is learnable |

Limiting cases the generator must support:

- `ν → 0` → near-copy frames (delta coding wins).  
- `ν` large or cut → keyframe-like residuals.  
- Single-modality dropout → remaining modalities still coherent with `S_t`.

---

## State variables (Markov test)

**Failed state:** raw pixels alone, redrawn → future independent of “present.”  

**Sufficient synthetic state `S_t`:**

1. Low-frequency background field `B` (fixed per clip)  
2. Blob parameters `{pos, radius, color, vel}_k`  
3. Ego velocity `(vx, vy)` (and optional yaw rate)  
4. Audio phase / envelope seed  

Given `S_t`, next frame is determined up to small sensor noise. History summary needed for the *codec* is still fused latents; for the *data*, `S_t` restores Markovianity.

---

## Optimization view (data design)

| Ingredient | Choice |
|------------|--------|
| Decision | How to sample `S_0`, controls `u_t`, noise levels |
| Objective | Maximize *useful* RD learning: structure that makes D and R move for the right reasons |
| Hard | Shared ego-motion across camera/LiDAR/IMU; values in valid ranges |
| Soft | Blob count, texture richness, cut probability |

Proxy objective (measurable without a human label):

\[
J_{\text{data}} = \mathbb{E}\big[\mathbf{1}[\phi_{\text{smooth}} < \phi_{\text{cut}}]\big]
\]

plus nonzero spatial gradient energy so MSE/PSNR can move.

---

## Conceptual model

**Category:** discrete-time stochastic process on a low-dimensional scene state, rendered to multimodal observations.

Assembly order:

1. Sample clip-level `B` + blobs.  
2. Each step: update ego + blob positions; render RGB.  
3. Derive LiDAR as noisy depth samples from blob/ground geometry.  
4. Derive IMU from ego velocity differences.  
5. Derive audio as slow envelope + weak coupling to motion magnitude.  
6. Rare Bernoulli cuts → resample `B`/blobs (adversarial temporal case).

---

## Failed guesses

1. **i.i.d. pixels = images** → train D fell (memorizing noise stats) while perceptual/eval PSNR stayed useless.  
2. **Additive `0.02 * step` on noise** → fake “motion” without spatial correlation; residual coding still starved.  
3. **Independent modality noise** → fusion cannot learn binding; cross-attention becomes decoration.

---

## Experiments before solutions

| Probe | Noise data | Structured data (required) |
|-------|------------|----------------------------|
| `φ` smooth vs cut | ≈ equal | cut ≫ smooth |
| Spatial grad energy | ~O(1) white | lower, edge-dominated |
| Eval PSNR after short train | stuck ~9 dB | must rise vs epoch-0 baseline |

---

## Domain of validity

This synthetic world is **not** real flight footage. It is a minimal world that respects temporal and cross-modal invariants so the codec’s math is not fighting the data. Real logs replace the renderer later without changing the RD objective.

---

## Implementation map

| Deliverable | Location |
|-------------|----------|
| Generator | `scripts/structured_synthetic_data.py` |
| Train wiring | `scripts/train_rd_compressor.py` |
| Eval wiring | `scripts/eval_compression_rd.py` |
| Resume note | Restart full train on structured clips; old noise weights are not comparable for PSNR |

**Pickup:** after restart, watch `φ` in logs (optional) and eval PSNR vs the noise-trained ~9 dB baseline.
