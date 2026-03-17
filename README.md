# Falcon H1 Path-Asymmetric LoRA — Experiment Code

Experiment code for the paper:
**"Path-Asymmetric Low-Rank Adaptation of Hybrid SSM-Transformer Models:
An Empirical Study on Falcon H1"**

---

## File structure

```
falcon_h1_experiments/
├── shared_utils.py       Common utilities (PEFT helpers, data loaders,
│                         metrics, plot helpers, custom Trainer)
├── lm_symmetric.py       Strategy S1: Symmetric LoRA on WikiText-2
├── lm_asymmetric.py      Strategies S2/S3: Asymmetric + Flipped LoRA
│                         on WikiText-2 (--rank_m / --rank_a flags)
├── tooluse_finetune.py   Fine-tuning on xLAM function-calling
│                         (--strategy symmetric | asymmetric | flipped)
├── tooluse_eval.py       Standalone evaluation on xLAM
│                         (--checkpoint or --zero_shot)
└── README.md             This file
```

---

## Requirements

```bash
pip install transformers peft datasets torch accelerate \
            bitsandbytes matplotlib seaborn
huggingface-cli login   # required for xLAM dataset access
```

---

## Reproducing the paper results

### Step 1 — Language modelling experiments

```bash
# S1: Symmetric LoRA (baseline)
python lm_symmetric.py --rank 16 --output_dir ./outputs/lm_symmetric

# S2: Asymmetric LoRA — Mamba-high (paper main result)
python lm_asymmetric.py --rank_m 16 --rank_a 4 --output_dir ./outputs/lm_asymmetric

# S3: Flipped LoRA — Attention-high (ablation)
python lm_asymmetric.py --rank_m 4 --rank_a 16 --output_dir ./outputs/lm_flipped
```

Expected results (WikiText-2 eval PPL):

| Strategy       | PPL  | Params (M) | ρ avg |
|----------------|------|------------|-------|
| S1 Symmetric   | 6.84 | 16.24      | 3.30  |
| S2 AsymLoRA    | 6.74 | 4.39       | 8.45  |
| S3 Flipped     | 6.81 | 4.97       | 0.80  |

### Step 2 — Tool-use fine-tuning

```bash
# S1
python tooluse_finetune.py --strategy symmetric

# S2
python tooluse_finetune.py --strategy asymmetric

# S3
python tooluse_finetune.py --strategy flipped
```

### Step 3 — Tool-use evaluation

```bash
# Zero-shot baseline
python tooluse_eval.py --zero_shot

# Fine-tuned checkpoints
python tooluse_eval.py --checkpoint ./outputs/tu_symmetric
python tooluse_eval.py --checkpoint ./outputs/tu_asymmetric
python tooluse_eval.py --checkpoint ./outputs/tu_flipped

# Cross-strategy comparison figure
python tooluse_eval.py --compare --results_root ./outputs
```

Expected results (xLAM Full-EM, 200 eval examples):

| Strategy     | FN-EM | PK    | PV    | Full-EM | Inv-JSON |
|--------------|-------|-------|-------|---------|----------|
| Zero-shot    | 0.220 | 0.235 | 0.150 | 0.105   | 0.615    |
| S1 Symmetric | 0.195 | 0.185 | 0.175 | 0.170   | 0.735    |
| S2 AsymLoRA  | 0.255 | 0.245 | 0.230 | 0.220   | 0.670    |
| S3 Flipped   | 0.320 | 0.300 | 0.280 | 0.275   | 0.610    |

---

## Key design notes

### PEFT blacklist
The PEFT library blocks `out_proj` and `conv1d` as LoRA targets for
`model_type=falcon_h1`. `shared_utils.py` enforces this via
`PEFT_BLACKLIST` before any adapter injection. The safe Mamba target
is `mamba.in_proj` (2048 → 6704). `mamba.out_proj` (3072 → 2048)
cannot be adapted via standard LoRA.

### No learned gate
Falcon H1 combines path outputs by addition (not a softmax gate).
See Equation (18) in the paper.

### Label masking for tool-use
Training loss is computed only on `<calls>...</calls>` completion
tokens. System prompt, tool schemas, and query are masked with
`label=-100` in `shared_utils.load_xlam`.

### extract_calls fix
`tooluse_eval.py::extract_calls` fixes the `IndexError` from the
original `tool_use_finetuning.py`. The regex fallback patterns for
bare JSON arrays/objects have no capture group; the fix uses
`.group(0)` instead of `.group(1)` for these patterns.

### Gradient norm ratio ρ
`shared_utils.compute_grad_ratio` computes ρ = ‖∇_mamba‖_F /
(‖∇_attn‖_F + ε) from the current gradient buffers. It is called
at every logging step inside `LoggingTrainer.training_step`.
ρ is informative post-hoc but should not be used as a pre-hoc
predictor of optimal rank direction (it is itself rank-modulated).

---

## Hyperparameters (paper values)

| Parameter           | LM experiments | Tool-use experiments |
|---------------------|----------------|----------------------|
| Epochs              | 3              | 3                    |
| Batch size          | 4              | 2                    |
| Grad accumulation   | 4              | 8                    |
| Effective batch     | 16             | 16                   |
| Learning rate       | 2e-4           | 2e-4                 |
| LR schedule         | Cosine         | Cosine               |
| Warmup ratio        | 0.03           | 0.03                 |
| Max seq length      | 512            | 1024                 |
| Train samples       | 2000           | 2000                 |
| Eval samples        | 200            | 200                  |
| LoRA dropout        | 0.05           | 0.05                 |
| Quantisation        | NF4 4-bit      | NF4 4-bit            |
