# True Rate vs Entropy Proxy — Applied-Math Notes

**Status:** active  
**Problem:** full train drove differentiable proxy `R → ~0.16` while PSNR held (~17.6 dB).  
**Claim to avoid:** “we only need 0.16 bits on the wire.”

Related: [RD_STABILITY_APPLIED_MATH.md](RD_STABILITY_APPLIED_MATH.md), [NEURAL_COMPRESSION_RD_PLAN.md](NEURAL_COMPRESSION_RD_PLAN.md)

---

## System (plain language)

The trainer scores “how surprising the codes look” with a learned probability model. That score can go near zero if the model becomes very sure about which symbols it will pick. Separately, the device still has to write those symbols into a packet. Until we run a real arithmetic/ANS coder that exploits that confidence, the packet size is set by how we store the symbols — usually one byte each when there are 256 possibilities — not by the surprise score.

---

## Inventory

| Entity | Role |
|--------|------|
| Continuous latent `z` | Pre-quant values |
| Index `q ∈ {0..L−1}^d` | What must be recoverable on the decoder |
| Entropy proxy `R_proxy` | Differentiable cross-entropy / model surprise (bits) |
| Fixed-length rate `R_fix` | `d · ⌈log₂ L⌉` bits |
| Packed payload | Bytes actually serialized (`d` bytes if `L ≤ 256`) |
| Countable rate `R_true` | `8 · packed_bytes` per sample (no ANS yet) |

---

## Failed guess

**Guess:** minimizing `R_proxy` is the same as minimizing uplink bits.  

**Break:** a peaked categorical can have tiny cross-entropy while each symbol still occupies a full byte in a naive packer. Proxy → 0, wire unchanged at `R_fix`.

---

## Invariants / bounds

| Kind | Statement |
|------|-----------|
| Bounded | `0 ≤ R_proxy` (ideally); in practice ≥ 0 after clamps |
| Lower bound (with ANS, ideal) | `R_wire ≥ H(q)` ≈ `R_proxy` in expectation if the model is calibrated |
| Lower bound (no ANS) | `R_true = R_fix` for fixed-length codes |
| Structural | Decoder needs `q` (or an equivalent bitstream), not `R_proxy` |

**Dimensionless:** `ρ = R_proxy / R_true`. Values `ρ ≪ 1` flag proxy collapse / missing entropy coder — exactly the post-train observation.

---

## Conceptual model

1. **Train** with `R_proxy` (differentiable).  
2. **Report** `R_true` and `R_fix` every eval.  
3. **Ship** packed indices (or ANS later).  
4. Only claim link bitrate from `R_true` (or ANS length).

---

## Implementation

| Piece | Location |
|-------|----------|
| Helpers | `lydlr_ai/model/true_rate.py` |
| Indices on forward | `EnhancedMultimodalCompressor` |
| Eval fields | `scripts/eval_compression_rd.py` |
| Orin path | `docs/guides/JETSON_ORIN_DEMO.md` |

**Defaults:** `L = 256`, `d = latent_dim` (64) → `R_fix = 512` bits = 64 bytes/sample.
