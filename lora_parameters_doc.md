LoRA has a small number of parameters but understanding what each one actually does — and what the downstream consequences are — takes some unpacking.

---

## The core parameters

**`r` — the rank**

This is the most important parameter. It controls the dimensionality of the low-rank decomposition. When you adapt a weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA replaces the full $d \times k$ update with two smaller matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, so the update lives in a subspace of dimension $r$.

Concretely for Falcon H1: adapting `in_proj` (2048×6704) with rank 16 costs $16 \times (2048 + 6704) = 140{,}032$ parameters instead of $2048 \times 6704 = 13{,}729{,}792$. You are betting that the useful part of the update lives in a 16-dimensional subspace of a 2048-dimensional space. This is the intrinsic dimensionality hypothesis — it holds well for most fine-tuning tasks but the optimal value of $r$ is task-dependent.

**Practical effect of increasing `r`:**
- More capacity to represent complex weight updates
- More trainable parameters (linear in $r$)
- Higher risk of overfitting on small datasets
- Higher gradient norm (more parameters accumulating signal)
- Diminishing returns: the useful update subspace for most tasks is low-dimensional, so going from rank 4 to rank 16 helps much more than going from rank 64 to rank 128

**`lora_alpha` — the scaling factor**

The adapted forward pass is $h = W_0 x + \frac{\alpha}{r} B A x$. The ratio $\alpha/r$ is a fixed scalar multiplier on the adapter's contribution. It is not trained — it is a hyperparameter that controls how large the adapter update is relative to the frozen base weights.

The key insight is that only the ratio $\alpha/r$ matters, not their individual values. Setting `alpha=32, r=16` gives $\alpha/r = 2.0$, and setting `alpha=16, r=16` gives $\alpha/r = 1.0$. The convention in the original LoRA paper is to set `alpha = r` (ratio = 1.0), but in practice many implementations use `alpha = 2r` because it keeps the adapter contribution at a stable magnitude as you vary rank.

**Practical effect of changing `alpha/r`:**
- Higher ratio → larger adapter updates → faster adaptation but risk of destabilising the base model's knowledge
- Lower ratio → smaller updates → more conservative fine-tuning, better for small datasets
- The effective learning rate of the adapter scales with $\alpha/r$, so if you change `r` without adjusting `alpha`, you implicitly change the update magnitude. This is why in your experiments `alpha_m=32, r_m=16` and `alpha_a=8, r_a=4` both give ratio 2.0 — the effective update scale is consistent across paths despite different ranks.

**`lora_dropout`**

Applied to the input before the adapter: $h = W_0 x + \frac{\alpha}{r} B \cdot \text{dropout}(A x)$. Regularises the adapter during training, preventing individual rank dimensions from becoming over-specialised. Has minimal effect with small datasets (dropout rarely fires enough to matter) and more effect with large datasets or high rank. Typical values: 0.0–0.1.

---

## The interaction between rank and dataset size

This is the most practically important relationship. Rank determines how many independent directions of adaptation are available. If your dataset is small (your 2000 WikiText-2 examples), a high-rank adapter has more parameters than the data can reliably estimate — the excess rank dimensions will fit noise. This is why S2 (rank 4 on attention) outperforms S1 (rank 16 on attention) on WikiText-2 with only 2000 examples: the attention path does not need 16 dimensions of adaptation for this domain, and the extra parameters hurt rather than help.

The rule of thumb: $r$ should be roughly proportional to the amount of task-specific information the target layer needs to absorb, which is bounded by both the dataset size and the task complexity. For adapting a pre-trained model to a narrow domain with a small dataset, rank 4–16 is usually sufficient. For multi-task fine-tuning or very different target domains, rank 32–64 may be warranted.

---

## The relationship between rank, alpha, and gradient dynamics

This is what your $\rho$ measurements reveal. The gradient flowing back through a LoRA adapter is:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top \frac{\partial \mathcal{L}}{\partial h}, \qquad \frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^\top$$

The Frobenius norm of these gradients scales with both $\alpha/r$ and the number of parameters — higher rank means more entries in $A$ and $B$ all receiving gradient, so the aggregate gradient norm $\|\nabla_\phi \mathcal{L}\|_F$ grows with $r$ even if the per-parameter gradient is the same. This is exactly why $\rho$ was 8.45 in S2 and 0.80 in S3 — switching rank 16 from Mamba to attention shifted the gradient mass to the attention path, changing $\rho$ by a factor of ~10.

---

## Summary table

| Parameter | What it controls | Too low | Too high |
|---|---|---|---|
| `r` | Adapter capacity (subspace dim) | Under-fitting, misses complex updates | Over-fitting, waste of params |
| `alpha/r` | Update magnitude relative to base | Slow adaptation, base dominates | Destabilises pre-trained knowledge |
| `dropout` | Adapter regularisation | Over-specialised rank dims | Adapter learns too slowly |
| `target_modules` | Which layers are adapted | Misses key layers | Updates irrelevant layers, noise |

The most consequential decisions in order are: (1) which layers to target, (2) rank, (3) alpha/r ratio, (4) dropout.
