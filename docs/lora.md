# LoRA — Low-Rank Adaptation of the Improved Student

This document specifies the LoRA implementation in
[`src/sleep_staging/adaptation/lora.py`](../src/sleep_staging/adaptation/lora.py),
its mathematics, its interaction with the frozen base, the verification
guarantees, and what the current experiments do and do not show.

---

## 1. Motivation

NeuroSleep investigates compact sleep staging under a strict parameter budget
(99,477 parameters ≈ 400 KB FP32). The adaptation experiments ask a single
research question:

> How much five-stage sleep-staging performance can be retained when a
> compact CNN–GRU network is adapted to a target cohort using
> parameter-efficient low-rank updates instead of full fine-tuning?

Full fine-tuning updates all 99,477 parameters per adaptation run. LoRA
freezes the base and trains only small low-rank factor matrices, enabling
cheap per-cohort/per-device adaptation from a shared base checkpoint.

---

## 2. Mathematical Formulation

For a frozen weight matrix `W ∈ R^{d_out × d_in}`, LoRA (Hu et al., 2021)
replaces the layer's forward map with:

```
W' = W + ΔW,        ΔW = B A

A ∈ R^{r × d_in}    (trainable, initialized ~ N(0, 0.01²))
B ∈ R^{d_out × r}   (trainable, initialized to zero)

forward:  y = W x + (α/r) · B A x
```

Properties that follow directly:

| Element | Trainable? | Initialization | Role |
|---------|-----------|----------------|------|
| `W` (base weight) | **frozen** | from base checkpoint | preserves pretrained behavior |
| `A` | trainable | small random | injects rank-r input directions |
| `B` | trainable | zeros | outputs start at 0 ⇒ **ΔW = 0 at init** |
| scaling `α/r` | fixed | — | decouples update magnitude from rank |

Because `B = 0` at initialization, a freshly wrapped model is
**functionally identical** to the base model (up to dropout), and training
moves only `A` and `B`.

### Linear layers (`LoRALinear`)

Wraps `nn.Linear`. Adapter path computes
`dropout(x) @ A^T @ B^T · (α/r)` and adds it to the frozen output.

### Convolutional layers (`LoRAConv1d`)

Wraps `nn.Conv1d`. For the pointwise (`kernel_size=1`) convolutions used by
the Improved Student (`enc.0.pw`, `enc.1.pw`), the layer is mathematically a
linear map at each time position, and the adapter is applied as an exact
low-rank linear update per position (`ΔW` has shape `[d_out, d_in]`).
Non-1×1 kernels fall back to a low-rank 1×1 additive conv; all current
targets are 1×1, so only the exact path is exercised.

### What is NOT adapted (important)

`nn.GRU` cannot currently be wrapped — `apply_lora` only handles
`nn.Linear` and `nn.Conv1d`. The GRU holds **89,856 / 99,477 parameters
(90.3%)** of the model. Consequently, every LoRA configuration run so far
adapts at most the convolutional representation and/or the classifier while
the recurrent temporal model remains completely frozen. This makes the
existing LoRA experiments a specifically scoped question:

> Can a tiny adaptation of the convolutional representation and
> classification head compensate for a completely frozen recurrent temporal
> model?

Implementing LoRA for the GRU weight matrices (`weight_ih_l0`,
`weight_hh_l0`, ...) is planned (see §10 Limitations and
`configs/lora_target_ablation.yaml`).

---

## 3. Target Modules

Available targets in the Improved Student (`named_modules()` paths):

| Module path | Type | Shape | Role |
|-------------|------|-------|------|
| `head` | `nn.Linear` | 64 → 5 | classification head |
| `gab_proj` | `nn.Linear` | 8 → 16 | Gabor feature projection |
| `enc.0.pw` | `nn.Conv1d` | 16 → 32, k=1 | CNN pointwise projection, block 0 |
| `enc.1.pw` | `nn.Conv1d` | 32 → 32, k=1 | CNN pointwise projection, block 1 |

Trainable-parameter counts per target (rank `r`):

| Target | Params per rank unit |
|--------|---------------------|
| `head` | `r·64 + 5·r = 69r` |
| `enc.0.pw` | `r·16 + 32·r = 48r` |
| `enc.1.pw` | `r·32 + 32·r = 64r` |
| `gab_proj` | `r·8 + 16·r = 24r` |

Primary configuration (`enc.0.pw + enc.1.pw + head`, r=8):
`(48+64+69)·8 = 1,448` trainable parameters = **1.43%** of the model.

Head-only r=8: `69·8 = 552` parameters = **0.55%**.

---

## 4. Configuration

```python
from sleep_staging.adaptation.lora import LoRAConfig, apply_lora

lora_config = LoRAConfig(
    rank=8,                                  # r
    alpha=16,                                # α  (scaling = α/r = 2.0)
    dropout=0.05,                            # adapter-input dropout
    target_modules=["enc.0.pw", "enc.1.pw", "head"],
)
model = apply_lora(model, lora_config)
# trainable params: 1,448 || all params: 99,477 || trainable%: 1.43%
```

| Hyperparameter | Value (primary) | Meaning |
|----------------|-----------------|---------|
| rank `r` | 8 | bottleneck dimensionality of ΔW |
| alpha `α` | 16 | scales ΔW; with α=2r the update scale is 2.0 |
| dropout | 0.05 | applied to adapter input only |
| bias | none | no bias adaptation |

---

## 5. Frozen vs Trainable Parameters

After `apply_lora`:

```text
Total parameters        99,477 + 2·(adapter tensors)   (adapters add params)
Base (frozen)           99,477                         requires_grad = False
LoRA A/B (trainable)    rank-dependent (1,448 primary)  requires_grad = True
```

`apply_lora` freezes **everything** first, then re-enables gradients only on
`LoRALinear.lora_A/B` and `LoRAConv1d.lora_A/B`. `count_lora_parameters`
reports `{total, trainable, frozen, trainable_pct}`.

---

## 6. The Three Adaptation Regimes (Explicit Definitions)

All three regimes start from the **same base checkpoint** and are evaluated
on the same 10 subject-level folds:

| Regime | Base weights | Trainable params | What is optimized |
|--------|--------------|-----------------|-------------------|
| **Frozen** | frozen | 0 | nothing — evaluation only |
| **LoRA** | frozen | 1,448 (primary cfg) | low-rank A/B factors on selected projection + classifier layers |
| **Full fine-tuning** | unfrozen | 99,477 | all parameters |

Full fine-tuning here means **adaptation from the pretrained base checkpoint**
(all 99,477 parameters unfrozen), which is distinct from
**from-scratch training** (random initialization), used by the primary
benchmark. The initialization point must always be stated.

---

## 7. Adapter Save / Load

- `save_adapter(model, path)` writes `adapter_model.pt` (per-target A/B
  tensors) + `adapter_config.json` (rank, alpha, scaling, target list).
- `load_adapter(model, path)` restores A/B onto a model that already has
  LoRA wrappers applied (same rank/targets).
- Base weights are **not** included in an adapter; shipping an adapter
  requires the base checkpoint it was trained on.

---

## 8. Merge Procedure

Merging is not yet automated in code. For a merged deployment, fold the
adapter into each target manually:

```text
W_merged = W_base + (α/r) · B A          # Linear: [d_out, d_in]
conv1x1: W_merged[out, in, 1] = W_base[out, in, 0...k] + (α/r) · (B A)[out, in]
```

For the 1×1 convolutions used here, `B A` is exactly `[d_out, d_in]` and can
be stacked to `[d_out, d_in, 1]` and added to the base kernel. Because
current targets are all 1×1, merging is exact (no approximation).

---

## 9. Verification Guarantees (Methodological, Not Just Engineering)

The following invariants are asserted by tests (`tests/test_lora.py`,
`tests/test_lora_conv1d.py`) and `scripts/smoke_test_lora.py`:

1. **Base preservation:** with LoRA applied and adapters at init (`B=0`),
   outputs equal the base model's outputs (zero-init guarantee of ΔW).
2. **Frozen base:** after `apply_lora`, no non-adapter parameter has
   `requires_grad=True`.
3. **Only adapters change:** after training, base weights are bit-identical
   to the base checkpoint; only `lora_A/lora_B` tensors differ.
4. **Wrapping completeness:** `assert_lora_targets` raises if any requested
   target was not actually wrapped.
5. **Adapter round-trip:** save → load reproduces identical outputs.
6. **Conv1d equivalence:** for k=1 targets, the low-rank adapter path is
   equivalent to adding a low-rank `ΔW` to the frozen kernel.

---

## 10. Results

> **CONTAMINATION WARNING (quarantined):** the existing adaptation results in
> `results/100_subject_adaptation/` were produced with the 15-subject-era
> base checkpoint, whose training subjects overlap the evaluation folds
> (12 test + 3 validation). Absolute numbers below are inflated (~+2.5pp on
> the frozen baseline) and **must not be reported as generalization
> results**. They are reproduced here solely to document what was run.

Quarantined legacy numbers (seed 42, 10 folds; LoRA/Full-FT mean over
3 seeds, frozen metrics seed-invariant):

| Method | Trainable | Accuracy | κ | Macro F1 | MGm |
|--------|----------:|---------:|---:|---------:|----:|
| Frozen | 0 | 87.1% | 0.738 | 0.673 | 0.668 |
| LoRA r=8 head | 552 | 83.1% | 0.682 | 0.659 | 0.724 |
| LoRA r=8 CNN+Head | 1,448 | 83.6% | 0.693 | 0.674 | 0.744 |
| Full FT (from base) | 99,477 | 87.7% | 0.763 | 0.730 | 0.796 |

Scientifically honest reading of the quarantined data:

> The tested CNN+Head LoRA configuration reduces trainable parameters by
> 68.7× (1,448 vs 99,477) but produces lower accuracy and substantially
> lower macro-F1 (0.674 vs 0.730) than full fine-tuning. LoRA notably
> increased minority-class (N1/REM) recall relative to the frozen base,
> trading precision. Whether a rank/target configuration (including GRU
> targets) closes this gap is an open question — see ablation configs.

For the uncontaminated primary from-scratch benchmark, see
[`docs/results.md`](results.md).

---

## 11. Limitations

1. **GRU not adaptable** — 90.3% of the model is excluded from every LoRA
   configuration run so far. The headline "LoRA underperforms full FT" is a
   claim about CNN+Head LoRA only, not about LoRA generally.
2. **Single rank tested at cohort scale** — r=8 only (rank ablation in
   progress; r=2/4/16 configs defined).
3. **Contaminated base** — all existing fold results used a base checkpoint
   overlapping evaluation subjects (see §10 warning).
4. **No cross-dataset validation** — Sleep-EDF only.
5. **No merge-time latency benchmark** — unmerged inference adds two small
   matmuls per adapted layer.

---

## References

- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language
  Models.* ICLR 2022. arXiv:2106.09685.
