# RD Training Stability — Applied-Math Notes

**Status:** active  
**Symptom:** structured-data full train logged `D ≈ 5995` on epoch 2, then recovered (`D ≈ 0.37` on epoch 3).  
**Question:** what is actually true about the loss, independent of optimizer folklore?

Related: [NEURAL_COMPRESSION_RD_PLAN.md](NEURAL_COMPRESSION_RD_PLAN.md), [TRAINING_DATA_APPLIED_MATH.md](TRAINING_DATA_APPLIED_MATH.md)

---

## System (plain language)

The trainer scores two kinds of “badness”: how wrong the rebuilt picture is, and how surprising the codes are (how many bits they would take). A third term from the VAE asks how weird the continuous latent statistics are versus a simple prior. That third term is not automatically bounded. If the network makes its own variance estimate huge for one step, that weirdness score can jump by thousands while the picture error barely moves — and the run looks like “distortion exploded” even when pixels are fine.

---

## Inventory

| Entity | Role |
|--------|------|
| Reconstruction error | Mean squared pixel error on `[0,1]` images |
| Rate `R` | Entropy proxy in bits over the quantized latent |
| KL to prior | VAE regularizer on `(μ, log σ²)` |
| Lagrange weight `λ` | Soft trade between rate and distortion |
| `β` | Soft weight on KL inside “distortion” |

**Measurables + units**

| Symbol | Units |
|--------|-------|
| `D_rec = MSE(x̂, x)` | dimensionless (intensity²) |
| `R` | bits / latent vector |
| `KL` | nats (or bits) per latent dim (mean) |
| `λ` | 1/bit when multiplying `R` |
| `log σ²` | log-variance (dimensionless log of variance) |

**Hard constraints we need**

- Finite loss every step (no `Inf`/`NaN` in the graph)  
- Transmit path still uses STE-quantized codes  
- Pixel targets stay in `[0,1]`

**Soft (fold into objective, not silent clamps of the world)**

- Prefer not to burn capacity on KL spikes  
- Prefer `R` reported relative to a known ceiling

---

## Failed guess (recorded)

**Guess:** “`distortion = MSE + β·KL` is a well-behaved scalar.”  

**Break:** let `log σ² → +∞` on even one coordinate. Then `σ² = exp(log σ²)` and  
`KL ⊃ ½ σ²` grows without bound. One bad batch → `D ~ 10³–10⁶` while `MSE` stays `O(10⁻¹)`. Epoch 2 matches this pattern (rate still fell; only “D” exploded).

So the invariant we want is not “KL conserved,” it is **KL (and therefore D) must be bounded or tempered** for the optimization to remain a well-posed RD trade.

---

## Candidate invariants / bounds

| Kind | Statement | Use |
|------|-----------|-----|
| Bounded | Pixel MSE ∈ `[0, 1]` for inputs in `[0,1]` (actually ≤1 for mean of squares of diffs ≤1) | `D_rec` cannot explain a 6000 spike |
| Bounded (enforced) | `log σ² ∈ [ℓ_min, ℓ_max]` | Caps `exp(log σ²)` |
| Bounded (enforced) | `KL_mean ∈ [0, KL_max]` after free-bits | Caps contribution to `D` |
| Scaling | `R_max = d · log₂(L)` for latent dim `d`, codebook size `L` | Dimensionless `ρ_R = R / R_max ∈ [0, ∞)` typically ≤ few |
| Monotone (desired) | Raising `λ` should not increase measured `R` on average | Eval probe |

**Free-bits:** do not pay KL below a small floor `κ` (information already “spent”). Implementation: `KL ← max(KL, κ)` is the usual *encourage* form; for *stability against spikes* we need the dual: **`KL ← min(KL, KL_max)`** (and optionally still a soft floor). Spike control is the hard bound; free-bits floor is optional for under-regularization.

---

## Symmetries

- **Description symmetry of latent scale:** absolute `(μ, σ)` units are arbitrary until quantization. Unbounded `log σ` is an artifact of parameterization, not a physical rate. → Clamp / reparameterize.  
- **RD objective symmetry:** only the pair `(D_rec, R)` is the codec operating point; KL is an auxiliary. Logging must **not** bury `D_rec` inside an unbounded KL sum without separate meters.

---

## Dimensionless RD objective (what we optimize)

Let

\[
\tilde{R} = \frac{R}{R_{\max}}, \quad R_{\max} = d\,\log_2 L,
\]

\[
\mathrm{KL}_{\bullet} = \mathrm{clip}\big(\mathrm{KL}(\mu,\log\sigma^2),\, 0,\, \mathrm{KL}_{\max}\big)
\quad\text{with}\quad \log\sigma^2 \in [\ell_{\min},\ell_{\max}],
\]

\[
\mathcal{L} = D_{\mathrm{rec}} + \beta\,\mathrm{KL}_{\bullet} + \lambda R + \text{(small consistency terms)}.
\]

Limiting cases:

| Limit | Required behavior |
|-------|-------------------|
| `ℓ_max → ∞` | recovers unstable loss (forbidden) |
| `KL_max → 0` | drops VAE regularizer (allowed ablation) |
| `λ → 0` | minimize distortion only |
| Static structured scene | `φ` small; `R` should fall after warm-start |

---

## State variables for the *optimizer* (Markov test)

Raw loss `L_t` alone is not a sufficient training state: two histories with the same `L` can differ by whether KL or MSE dominates.

**Sufficient logged state per epoch:**

1. `D_rec`, `KL_•`, `R`, `ρ_R = R/R_max`  
2. `φ` (data residual)  
3. Fraction of steps that hit the KL cap (should be rare after warm-up)

If KL-cap fraction stays high, the model is still fighting the prior — fix architecture/β, do not raise `KL_max`.

---

## Conceptual fix (category first)

**Category:** constrained / tempered variational RD loss — not “add another fudge weight.”

Assembly:

1. Clamp `logvar` (and lightly `μ`) at encode or loss.  
2. Compute mean KL; clip to `[0, KL_max]`.  
3. Keep `D_rec` and `KL` as separate metrics forever.  
4. Skip non-finite steps (hard constraint).  
5. Keep existing grad clip as a second safety layer.

---

## Domain of validity

- Caps hide pathology if `KL_max` is hit every step — watch the hit rate.  
- Does not fix wrong data (noise) — that was a separate failed guess.  
- Entropy `R` remains a proxy until a real coder is attached.

---

## Implementation

| Piece | Location |
|-------|----------|
| Math (this doc) | `docs/architecture/RD_STABILITY_APPLIED_MATH.md` |
| Loss tempering | `compute_rd_loss` in `compressor.py` |
| Non-finite skip | `scripts/train_rd_compressor.py` |

**Constants (measurable defaults):**  
`ℓ ∈ [-8, 8]`, `KL_max = 20` (mean nats), `L = 256`, `d = latent_dim`.

---

## Plausibility

- Epoch-2 spike with falling `R` ⇒ not a rate bug; matches unbounded KL.  
- Recovery on epoch 3 ⇒ intermittent; clamps prevent the intermittent catastrophe from dominating Adam moments as hard.
