# LoRA Rank and PEFT Parameters: A Technical Guide

## Overview

LoRA (Low-Rank Adaptation) has a small number of hyperparameters but
understanding what each one actually does — and the downstream consequences
on training dynamics, parameter efficiency, and task performance — takes
careful unpacking. This guide covers each parameter in depth, their
interactions, and practical guidelines derived from the Falcon H1
asymmetric fine-tuning experiments.

---

## 1. The Core Decomposition

For a frozen pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$,
LoRA constrains the update to a low-rank subspace:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$,
and $r \ll \min(d, k)$.

The modified forward pass is:

$$h = W_0 x + \frac{\alpha}{r} B A x = h_0 + \frac{\alpha}{r} B (Ax)$$

At initialisation: $A \sim \mathcal{N}(0, \sigma^2)$, $B = \mathbf{0}$,
so $\Delta W = \mathbf{0}$ at step zero. The fine-tuned model is
identical to the base model at the start of training.

Only $A$ and $B$ are trained. $W_0$ is frozen.

---

## 2. `r` — The Rank

### What it controls

$r$ is the dimensionality of the low-rank decomposition. It determines
how many independent directions of adaptation are available. The update
$\Delta W = \frac{\alpha}{r} BA$ lives in a subspace of dimension at
most $r$ within the full $d \times k$ parameter space.

You are betting that the useful part of the weight update for your target
task lies in a low-dimensional subspace of the full parameter space. This
is the **intrinsic dimensionality hypothesis** — it holds well for most
fine-tuning tasks, and the optimal $r$ is task-dependent.

### Parameter cost

$$|\phi| = r(d + k)$$

versus $dk$ for full fine-tuning. For Falcon H1's `mamba.in_proj`
(2048 → 6704) with rank 16:

$$16 \times (2048 + 6704) = 140{,}032 \text{ parameters}$$

versus $2048 \times 6704 = 13{,}729{,}792$ for full fine-tuning.
A **98× reduction**.

### Practical effects of increasing `r`

| Effect | Direction | Explanation |
|--------|-----------|-------------|
| Trainable parameters | ↑ linear in r | More $B$/$A$ entries |
| Adapter capacity | ↑ | More independent update directions |
| Overfitting risk | ↑ | Excess rank dims fit noise on small datasets |
| Gradient norm $\|\nabla_\phi \mathcal{L}\|_F$ | ↑ | More params accumulate signal |
| Diminishing returns | ↑ | Useful subspace is bounded by task complexity |

### Observed in Falcon H1 experiments

Going from rank 4 (attention) to rank 16 (attention) in the flipped
configuration shifted $\rho$ from 8.45 → 0.80, confirming that rank
directly controls gradient mass concentration.

### Practical guideline

- **Small dataset, narrow domain** (2,000 examples): rank 4–16 is
  usually sufficient. The useful update subspace is small.
- **Large dataset, diverse domain**: rank 32–64 may be warranted.
- **Diminishing returns**: improvements from rank 4→16 are much larger
  than from rank 64→128. Beyond rank 64, gains are typically marginal.
- **Always validate rank direction** with a cheap ablation before a full
  run. The optimal rank and direction are task-dependent.

---

## 3. `lora_alpha` — The Scaling Factor

### What it controls

$\alpha$ is a fixed scalar hyperparameter — it is **not trained**. It
controls the magnitude of the adapter's contribution to the forward pass
via the ratio $\alpha/r$.

Only the **ratio** $\alpha/r$ matters, not their individual values:

| `alpha` | `r` | $\alpha/r$ | Effective update scale |
|---------|-----|------------|----------------------|
| 16      | 16  | 1.0        | Conservative         |
| 32      | 16  | 2.0        | Moderate (common)    |
| 64      | 16  | 4.0        | Aggressive           |
| 8       | 4   | 2.0        | Same as row 2        |

### Why the ratio matters

The adapter update $\frac{\alpha}{r} BA x$ is added to the frozen
base output $W_0 x$. The ratio $\alpha/r$ determines how large this
addition is relative to the base contribution. If $\alpha/r$ is too
large, the adapter can destabilise the pre-trained knowledge encoded
in $W_0$. If it is too small, adaptation is slow and the base model
dominates throughout training.

### Keeping the ratio consistent across ranks

When using asymmetric ranks (e.g. $r_m = 16$, $r_a = 4$), set alphas
proportionally so the effective update scale is the same on both paths:

```python
alpha_m = 2 * rank_m   # alpha_m=32, r_m=16  →  ratio=2.0
alpha_a = 2 * rank_a   # alpha_a=8,  r_a=4   →  ratio=2.0
```

This ensures the difference in gradient dynamics between paths reflects
the rank difference, not an inadvertent difference in update magnitude.

### Practical guideline

- Set $\alpha = 2r$ as the default (ratio = 2.0). This is the most
  common convention and works well across most tasks.
- Use $\alpha = r$ (ratio = 1.0) for more conservative fine-tuning
  on very small datasets.
- Never change `alpha` without considering its effect on $\alpha/r$.
  Doubling `alpha` is equivalent to doubling the effective learning rate
  of the adapter.

---

## 4. `lora_dropout`

### What it controls

Dropout applied to the input of the low-rank adapter before computing
$Ax$:

$$h = W_0 x + \frac{\alpha}{r} B \cdot \text{dropout}(Ax)$$

Regularises the adapter during training by randomly zeroing elements,
preventing individual rank dimensions from over-specialising.

### Practical effects

| Dataset size | Recommended dropout | Reasoning |
|--------------|--------------------|-----------| 
| Small (<5k)  | 0.0–0.05           | Dropout rarely fires enough to matter; model needs all gradient signal |
| Medium (5k–50k) | 0.05–0.10       | Mild regularisation helps generalisation |
| Large (>50k) | 0.10               | Stronger regularisation, especially with high rank |

In the Falcon H1 experiments (2,000 training examples), `dropout=0.05`
was used. At this dataset size, dropout has minimal effect on the final
result but does not hurt.

---

## 5. `target_modules`

### What it controls

Which linear layers receive LoRA adapters. This is the **most impactful
decision** in PEFT configuration — more consequential than rank choice.

### PEFT leaf-name matching

PEFT matches `target_modules` using suffix regex:

```python
re.match(rf".*\.{key}$", module_full_path)
```

A key `"q_proj"` matches any module whose full path ends in `.q_proj`,
e.g. `"layers.3.self_attn.q_proj"`. This means:

- Leaf names work correctly as keys
- You cannot target a specific layer's `q_proj` without targeting all
  layers' `q_proj` (unless you use full dotted paths as keys)
- For asymmetric per-path targeting, use separate keys per path type

### PEFT blacklist for Falcon H1

The PEFT library blocks `out_proj` and `conv1d` for
`model_type=falcon_h1`. Including either in `target_modules` raises
a `ValueError`:

```
ValueError: [PEFT:LORA] Module 'out_proj' is incompatible with
Mamba-based models (model_type='falcon_h1').
```

Safe targets for Falcon H1:

| Layer | Path | Safe? |
|-------|------|-------|
| `mamba.in_proj` | SSM input expansion (2048→6704) | ✓ |
| `mamba.out_proj` | SSM output projection (3072→2048) | ✗ Blocked |
| `self_attn.q_proj` | Query projection (2048→1024) | ✓ |
| `self_attn.k_proj` | Key projection (2048→256) | ✓ |
| `self_attn.v_proj` | Value projection (2048→256) | ✓ |
| `self_attn.o_proj` | Output projection (1024→2048) | ✓ |
| `feed_forward.*` | Gate/up/down projections | ✓ |

### Practical guideline

Adapt layers where gradient signal concentrates. Adapting layers
that need minimal change introduces noise without benefit — as observed
when Symmetric LoRA degraded FN-EM below zero-shot on the tool-use task
by unnecessarily updating all attention projections at high rank.

---

## 6. `rank_pattern` and `alpha_pattern`

### What they control

Per-module overrides for rank and alpha. When provided, PEFT checks each
targeted module against the pattern dict before applying the base `r`.
This is how asymmetric LoRA is implemented in a single `LoraConfig`.

```python
LoraConfig(
    r=16,                         # base rank (fallback)
    lora_alpha=32,                # base alpha (fallback)
    target_modules=["in_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
    rank_pattern={
        "in_proj": 16,            # Mamba path → rank 16
        "q_proj":   4,            # Attention path → rank 4
        "k_proj":   4,
        "v_proj":   4,
        "o_proj":   4,
    },
    alpha_pattern={
        "in_proj": 32,            # alpha/r = 2.0 on both paths
        "q_proj":   8,
        "k_proj":   8,
        "v_proj":   8,
        "o_proj":   8,
    },
)
```

### Silent fallback risk

If a module's leaf name is **not** found in `rank_pattern`, PEFT falls
back to the base `r`. This can silently produce a different rank than
intended if keys are misspelled or a new layer type is added without
updating the pattern. Always verify with:

```python
for name, param in model.named_parameters():
    if "lora_A" in name:
        # lora_A shape is (r, d_in) — shape[0] is the rank
        print(name, param.shape[0])
```

---

## 7. The Rank–Dataset–Capacity Triangle

The three quantities that must be balanced:

```
     Task complexity
           /\
          /  \
         /    \
        /      \
  Dataset  ——  Rank
    size
```

- **Rank too high for dataset size**: adapter overfits. The excess rank
  dimensions capture noise specific to the training examples. Val loss
  will diverge from train loss.
- **Rank too low for task complexity**: adapter underfits. The update
  subspace is too small to represent the required weight changes.
  Val loss plateaus early.
- **Balanced**: adapter captures the true task-specific subspace. Val
  loss tracks train loss closely and continues improving.

### Observed in Falcon H1 experiments

| Config | Rank (attn) | Dataset | PPL | Interpretation |
|--------|-------------|---------|-----|----------------|
| S1 Symmetric | 16 | 2,000 | 6.84 | High rank on attention, marginal for this domain |
| S2 AsymLoRA  | 4  | 2,000 | 6.74 | Attention rank matched to its adaptation need |
| S3 Flipped   | 16 | 2,000 | 6.81 | High attention rank, helpful but Mamba rank too low |

---

## 8. The Gradient Norm Ratio $\rho$

Defined as:

$$\rho = \frac{\|\nabla_{\Theta_m} \mathcal{L}\|_F}{\|\nabla_{\Theta_a} \mathcal{L}\|_F + \varepsilon}$$

$\rho > 1$ means the Mamba path's adapters accumulate more gradient
signal. The gradient norm of a LoRA adapter scales with both $\alpha/r$
and the number of parameters, so:

$$\|\nabla_\phi \mathcal{L}\|_F \propto r \cdot \frac{\alpha}{r} = \alpha$$

Higher rank on a path → more parameters → larger aggregate gradient norm
→ higher $\rho$ for that path. This is why $\rho$ is **rank-modulated**
and cannot be used as a pre-hoc predictor of the optimal rank direction
without testing both configurations.

| Strategy | $r_m$ | $r_a$ | $\rho_\text{avg}$ |
|----------|--------|--------|-------------------|
| S2 AsymLoRA | 16 | 4  | 8.45 |
| S3 Flipped  | 4  | 16 | 0.80 |
| S1 Symmetric| 16 | 16 | 3.30 |

$\rho$ is informative **post-hoc** — it confirms which path received
more gradient mass under the chosen configuration — but it is circular
as a selection criterion.

---

## 9. Summary: Decision Priority

The most impactful decisions in order:

1. **`target_modules`** — Which layers to adapt. Wrong choice
   introduces noise or misses key update directions entirely.
2. **`r` (rank)** — How much capacity to give the adapter. Must be
   matched to dataset size and task complexity.
3. **`alpha/r` ratio** — Update magnitude. Use $\alpha = 2r$ as default.
   Keep consistent across paths when using asymmetric ranks.
4. **`lora_dropout`** — Regularisation. Only matters with larger datasets
   or higher ranks.

### Quick reference

```python
# Conservative (small dataset, narrow domain)
LoraConfig(r=4,  lora_alpha=8,  lora_dropout=0.0)

# Standard (moderate dataset)
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)

# Asymmetric (Mamba-high, Falcon H1 language modelling)
LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["in_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
    rank_pattern= {"in_proj": 16, "q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4},
    alpha_pattern={"in_proj": 32, "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8},
)

# Asymmetric (Attention-high, Falcon H1 tool use)
LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["in_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
    rank_pattern= {"in_proj": 4,  "q_proj": 16, "k_proj": 16, "v_proj": 16, "o_proj": 16},
    alpha_pattern={"in_proj": 8,  "q_proj": 32, "k_proj": 32, "v_proj": 32, "o_proj": 32},
)
```
