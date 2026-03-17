"""
Strategy 2: Path-Asymmetric LoRA for Falcon H1
================================================
Applies LoRA adapters with different ranks to the Mamba (fast) path
and the Transformer (slow) path. The Mamba path gets a higher rank
adapter to absorb most domain adaptation; the attention path gets a
minimal rank adapter (or is frozen) to stay cheap.

Paper metrics and plots produced at end of training:
  - train/eval loss curves                     → fig1_loss_curves.pdf
  - perplexity curve (train + eval)            → fig2_perplexity_curves.pdf
  - gradient norm ratio ρ + per-path norms     → fig3_gradient_norms.pdf
  - trainable parameter breakdown + comparison → fig4_param_breakdown.pdf
  - LoRA adapter weight magnitude heatmap      → fig5_adapter_heatmap.pdf
  - learning-rate schedule                     → fig6_lr_schedule.pdf
  - metrics_summary.json  ← machine-readable for LaTeX table

Requirements:
    pip install transformers peft datasets torch accelerate bitsandbytes matplotlib seaborn
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── consistent plot style ────────────────────────────────────────────────────
MAMBA_COLOR  = "#1f77b4"
ATTN_COLOR   = "#9467bd"
RATIO_COLOR  = "#ff7f0e"
TRAIN_COLOR  = "#2ca02c"
EVAL_COLOR   = "#d62728"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
    "savefig.dpi":       200,
})

def _save(fig, out_dir: Path, stem: str):
    """Save a figure as both PDF and PNG with tight bounding box."""
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # Model
    model_name: str   = "tiiuae/Falcon-H1-1B-Base"
    load_in_4bit: bool = True

    # LoRA ranks (the core asymmetry)
    mamba_lora_rank: int  = 4
    attn_lora_rank: int   = 16
    freeze_attn: bool     = False

    lora_alpha_mamba: int = 4
    lora_alpha_attn: int  = 32
    lora_dropout: float   = 0.05

    # Data
    dataset_name:   str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int          = 512
    num_train_samples: int   = 2000
    num_eval_samples: int    = 200

    # Training
    output_dir: str                    = "./outputs/asymmetric_lora_attn"
    num_train_epochs: int              = 3
    per_device_train_batch_size: int   = 8
    gradient_accumulation_steps: int   = 4
    learning_rate: float               = 2e-4
    warmup_ratio: float                = 0.03
    lr_scheduler_type: str             = "cosine"
    fp16: bool         = True
    logging_steps: int = 20
    eval_steps: int    = 100
    save_steps: int    = 200
    seed: int          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Module detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_PEFT_MAMBA_BLACKLIST = {"out_proj", "conv1d"}


def get_mamba_target_modules(model: nn.Module) -> list:
    """
    Return Linear leaf-names belonging to the Mamba path,
    excluding PEFT-blacklisted modules ('out_proj', 'conv1d').
    """
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        is_mamba = "mamba" in lower or any(
            p in lower for p in ["x_proj", "dt_proj", "in_proj", "out_proj"]
        )
        is_attn = "attn" in lower or "attention" in lower
        if is_mamba and not is_attn:
            leaf = name.split(".")[-1]
            if leaf in _PEFT_MAMBA_BLACKLIST:
                logger.info(f"  skip '{leaf}' (PEFT blacklist) — {name}")
                continue
            if leaf not in found:
                found.append(leaf)
    logger.info(f"Mamba target modules: {found}")
    return found or ["x_proj", "dt_proj", "in_proj"]


def get_attn_target_modules(model: nn.Module) -> list:
    """Return Linear leaf-names belonging to the attention path."""
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        if "attn" in lower or "attention" in lower:
            leaf = name.split(".")[-1]
            if leaf not in found:
                found.append(leaf)
    logger.info(f"Attention target modules: {found}")
    return found or ["q_proj", "k_proj", "v_proj", "o_proj"]


def print_model_linear_modules(model: nn.Module):
    """Debug: print every nn.Linear grouped by path. Run once before training."""
    mamba_l, attn_l, other_l = [], [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lower = name.lower()
            if "mamba" in lower or any(p in lower for p in ["x_proj","dt_proj","in_proj"]):
                mamba_l.append(name)
            elif "attn" in lower or "attention" in lower:
                attn_l.append(name)
            else:
                other_l.append(name)
    print("\n=== Mamba-path Linear layers ===")
    for n in mamba_l:  print("  ", n)
    print("\n=== Attention-path Linear layers ===")
    for n in attn_l:   print("  ", n)
    print("\n=== Other Linear layers ===")
    for n in other_l:  print("  ", n)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Asymmetric LoRA setup
# ─────────────────────────────────────────────────────────────────────────────

def apply_asymmetric_lora(model: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    mamba_targets = get_mamba_target_modules(model)
    attn_targets  = get_attn_target_modules(model)

    if cfg.freeze_attn:
        for name, param in model.named_parameters():
            if any(t in name for t in attn_targets):
                param.requires_grad = False
        logger.info("Attention path hard-frozen (strategy-1 mode).")
        target_modules = [m for m in mamba_targets if m not in _PEFT_MAMBA_BLACKLIST]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.mamba_lora_rank,
            lora_alpha=cfg.lora_alpha_mamba,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    else:
        all_targets   = list(set(mamba_targets + attn_targets))
        rank_pattern  = {t: cfg.mamba_lora_rank for t in mamba_targets}
        rank_pattern.update({t: cfg.attn_lora_rank for t in attn_targets})
        alpha_pattern = {t: cfg.lora_alpha_mamba  for t in mamba_targets}
        alpha_pattern.update({t: cfg.lora_alpha_attn for t in attn_targets})

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.mamba_lora_rank,
            lora_alpha=cfg.lora_alpha_mamba,
            lora_dropout=cfg.lora_dropout,
            target_modules=all_targets,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            bias="none",
        )
        logger.info(f"AsymLoRA  r_mamba={cfg.mamba_lora_rank}  r_attn={cfg.attn_lora_rank}")

    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-norm path monitor
# ─────────────────────────────────────────────────────────────────────────────

class GradientPathMonitor:
    """
    Backward hooks that track per-path gradient norms every logging step.
    Recorded quantities (used in paper Figure 3):
        mamba_norm_history  — mean ||grad|| over Mamba LoRA params
        attn_norm_history   — mean ||grad|| over Attn  LoRA params
        ratio_history       — ρ = mamba_norm / attn_norm
        steps               — corresponding global steps
    """

    def __init__(self, model: nn.Module):
        self.mamba_norm_history = []
        self.attn_norm_history  = []
        self.ratio_history      = []
        self.steps              = []
        self._hooks    = []
        self._m_buf    = []
        self._a_buf    = []
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lower = name.lower()
            if "mamba" in lower or any(p in lower for p in ["x_proj","dt_proj","in_proj"]):
                self._hooks.append(
                    param.register_hook(lambda g: self._m_buf.append(g.norm().item()))
                )
            elif "attn" in lower or "attention" in lower:
                self._hooks.append(
                    param.register_hook(lambda g: self._a_buf.append(g.norm().item()))
                )

    def step(self, global_step: int):
        mn = float(np.mean(self._m_buf)) if self._m_buf else 0.0
        an = float(np.mean(self._a_buf)) if self._a_buf else 0.0
        self.mamba_norm_history.append(mn)
        self.attn_norm_history.append(an)
        self.ratio_history.append(mn / (an + 1e-9))
        self.steps.append(global_step)
        self._m_buf.clear()
        self._a_buf.clear()

    def report(self) -> dict:
        if not self.mamba_norm_history:
            return {}
        avg_m = float(np.mean(self.mamba_norm_history))
        avg_a = float(np.mean(self.attn_norm_history))
        avg_r = float(np.mean(self.ratio_history))
        logger.info(
            f"[GradMonitor] Mamba={avg_m:.4f}  Attn={avg_a:.4f}  ratio={avg_r:.2f}"
        )
        return {
            "mamba_avg":   avg_m,
            "attn_avg":    avg_a,
            "ratio_avg":   avg_r,
            "ratio_final": self.ratio_history[-1],
        }

    def remove(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Trainer
# ─────────────────────────────────────────────────────────────────────────────

class AsymmetricLoRATrainer(Trainer):
    def __init__(self, *args, monitor: Optional[GradientPathMonitor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self._train_loss_log = []   # (step, loss)
        self._eval_loss_log  = []   # (step, loss)
        self._lr_log         = []   # (step, lr)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.monitor and self.state.global_step % self.args.logging_steps == 0:
            self.monitor.step(self.state.global_step)
        return loss

    def log(self, logs: dict, start_time=None):
        super().log(logs)
        step = self.state.global_step
        if "loss" in logs:
            self._train_loss_log.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self._eval_loss_log.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self._lr_log.append((step, logs["learning_rate"]))


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(cfg: ExperimentConfig, tokenizer):
    dataset    = load_dataset(cfg.dataset_name, cfg.dataset_config)
    train_data = dataset["train"].select(range(cfg.num_train_samples))
    eval_data  = dataset["validation"].select(range(cfg.num_eval_samples))

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])
    eval_data  = eval_data.map( tokenize, batched=True, remove_columns=["text"])
    train_data.set_format("torch")
    eval_data.set_format("torch")
    return train_data, eval_data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


def count_trainable_params(model: nn.Module) -> dict:
    mamba_p, attn_p, other_p, frozen_p = 0, 0, 0, 0
    for name, param in model.named_parameters():
        lower = name.lower()
        n = param.numel()
        if not param.requires_grad:
            frozen_p += n
        elif "mamba" in lower or any(p in lower for p in ["x_proj","dt_proj","in_proj"]):
            mamba_p += n
        elif "attn" in lower or "attention" in lower:
            attn_p += n
        else:
            other_p += n
    total = mamba_p + attn_p + other_p + frozen_p
    return {
        "mamba_trainable": mamba_p,
        "attn_trainable":  attn_p,
        "other_trainable": other_p,
        "frozen":          frozen_p,
        "total":           total,
        "pct_trainable":   round(100 * (mamba_p + attn_p + other_p) / max(total, 1), 2),
    }


def collect_adapter_magnitudes(model: nn.Module) -> dict:
    mamba_n, attn_n = [], []
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        lower = name.lower()
        norm  = param.data.float().norm().item()
        if "mamba" in lower or any(p in lower for p in ["x_proj","dt_proj","in_proj"]):
            mamba_n.append(norm)
        elif "attn" in lower or "attention" in lower:
            attn_n.append(norm)
    return {"mamba": mamba_n, "attn": attn_n}


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(train_log, eval_log, out_dir: Path):
    """Figure 1 — cross-entropy loss."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if train_log:
        s, v = zip(*train_log)
        ax.plot(s, v, color=TRAIN_COLOR, lw=1.8, label="Train loss")
    if eval_log:
        s, v = zip(*eval_log)
        ax.plot(s, v, color=EVAL_COLOR, lw=1.8, ls="--", marker="o", ms=4,
                label="Eval loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and evaluation loss — AsymLoRA")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig1_loss_curves")
    logger.info("Saved fig1_loss_curves")


def plot_perplexity_curves(train_log, eval_log, out_dir: Path):
    """Figure 2 — perplexity."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if train_log:
        s, v = zip(*train_log)
        ax.plot(s, [compute_perplexity(x) for x in v],
                color=TRAIN_COLOR, lw=1.8, label="Train PPL")
    if eval_log:
        s, v = zip(*eval_log)
        ax.plot(s, [compute_perplexity(x) for x in v],
                color=EVAL_COLOR, lw=1.8, ls="--", marker="o", ms=4,
                label="Eval PPL")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity curves — AsymLoRA")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig2_perplexity_curves")
    logger.info("Saved fig2_perplexity_curves")


def plot_gradient_norm_ratio(monitor: GradientPathMonitor, out_dir: Path):
    """Figure 3 — gradient norm ratio ρ and per-path norms."""
    if not monitor.ratio_history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(monitor.steps, monitor.ratio_history, color=RATIO_COLOR, lw=1.8)
    ax.axhline(1.0, color="gray", lw=1.0, ls="--", label="ρ = 1 (equal paths)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("ρ  (Mamba / Attn grad norm)")
    ax.set_title("Gradient norm ratio ρ")
    ax.legend()

    ax = axes[1]
    if monitor.mamba_norm_history:
        ax.plot(monitor.steps[:len(monitor.mamba_norm_history)],
                monitor.mamba_norm_history,
                color=MAMBA_COLOR, lw=1.5, label="Mamba path")
    if monitor.attn_norm_history:
        ax.plot(monitor.steps[:len(monitor.attn_norm_history)],
                monitor.attn_norm_history,
                color=ATTN_COLOR, lw=1.5, ls="--", label="Attn path")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean gradient norm")
    ax.set_title("Per-path gradient norms")
    ax.legend()

    fig.suptitle("Gradient analysis — AsymLoRA", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_gradient_norms")
    logger.info("Saved fig3_gradient_norms")


def plot_param_breakdown(param_counts: dict, cfg: ExperimentConfig, out_dir: Path):
    """Figure 4 — trainable parameter pie + symmetric vs asymmetric bar."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # pie
    ax = axes[0]
    labels = ["Mamba LoRA", "Attn LoRA", "Other trainable", "Frozen"]
    sizes  = [param_counts["mamba_trainable"], param_counts["attn_trainable"],
              param_counts["other_trainable"],  param_counts["frozen"]]
    colors = [MAMBA_COLOR, ATTN_COLOR, TRAIN_COLOR, "#cccccc"]
    nz = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if nz:
        l2, s2, c2 = zip(*nz)
        _, _, auts = ax.pie(
            s2, labels=l2, colors=c2, autopct="%1.1f%%", startangle=90,
            pctdistance=0.82, wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in auts:
            at.set_fontsize(9)
    ax.set_title(f"Parameter breakdown\n({param_counts['pct_trainable']:.1f}% trainable)")

    # bar: symmetric vs asymmetric
    ax = axes[1]
    sym_mamba = param_counts["mamba_trainable"] * (16 / max(cfg.mamba_lora_rank, 1))
    sym_attn  = param_counts["attn_trainable"]  * (16 / max(cfg.attn_lora_rank,  1))
    strategies  = ["Symmetric\nLoRA (r=16)", "Asymmetric\nLoRA (ours)"]
    mamba_vals  = [sym_mamba / 1e6, param_counts["mamba_trainable"] / 1e6]
    attn_vals   = [sym_attn  / 1e6, param_counts["attn_trainable"]  / 1e6]
    x, w = np.arange(2), 0.35
    b1 = ax.bar(x - w/2, mamba_vals, w, label="Mamba LoRA", color=MAMBA_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, attn_vals,  w, label="Attn LoRA",  color=ATTN_COLOR,  alpha=0.85)
    ax.set_ylabel("Trainable params (M)")
    ax.set_title("Parameter budget: symmetric vs asymmetric")
    ax.set_xticks(x); ax.set_xticklabels(strategies)
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.1f}M", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Trainable parameter analysis — AsymLoRA", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig4_param_breakdown")
    logger.info("Saved fig4_param_breakdown")


def plot_adapter_magnitude_heatmap(magnitudes: dict, out_dir: Path):
    """Figure 5 — LoRA adapter weight Frobenius norm per layer."""
    mamba_n = magnitudes.get("mamba", [])
    attn_n  = magnitudes.get("attn",  [])
    if not mamba_n and not attn_n:
        return
    n_rows = max(len(mamba_n), len(attn_n), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, n_rows * 0.28 + 1)))

    def _heatmap(ax, norms, title, cmap):
        if not norms:
            ax.set_visible(False)
            return
        data = np.array(norms).reshape(-1, 1)
        sns.heatmap(
            data, ax=ax, cmap=cmap, annot=True, fmt=".3f",
            xticklabels=["||W||_F"],
            yticklabels=[f"Layer {i}" for i in range(len(norms))],
            linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)

    _heatmap(axes[0], mamba_n, "Mamba LoRA adapter norms", "Blues")
    _heatmap(axes[1], attn_n,  "Attn LoRA adapter norms",  "Purples")
    fig.suptitle("LoRA adapter weight magnitudes — AsymLoRA", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_adapter_heatmap")
    logger.info("Saved fig5_adapter_heatmap")


def plot_lr_schedule(lr_log, out_dir: Path):
    """Figure 6 — learning-rate schedule."""
    if not lr_log:
        return
    steps, lrs = zip(*lr_log)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(steps, lrs, color="#17becf", lw=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedule (cosine with warmup) — AsymLoRA")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    fig.tight_layout()
    _save(fig, out_dir, "fig6_lr_schedule")
    logger.info("Saved fig6_lr_schedule")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_json(cfg, param_counts, grad_stats, trainer_state, out_dir: Path):
    """
    Dump all scalar metrics to metrics_summary.json.
    All values are JSON-serialisable — ready for the LaTeX table.
    """
    best_eval_loss, best_eval_ppl = None, None
    if trainer_state and trainer_state.log_history:
        evals = [e for e in trainer_state.log_history if "eval_loss" in e]
        if evals:
            best_eval_loss = min(e["eval_loss"] for e in evals)
            best_eval_ppl  = compute_perplexity(best_eval_loss)

    summary = {
        "strategy":          "AsymmetricLoRA",
        "model":             cfg.model_name,
        "mamba_lora_rank":   cfg.mamba_lora_rank,
        "attn_lora_rank":    cfg.attn_lora_rank,
        "freeze_attn":       cfg.freeze_attn,
        "mamba_trainable_M": round(param_counts["mamba_trainable"] / 1e6, 3),
        "attn_trainable_M":  round(param_counts["attn_trainable"]  / 1e6, 3),
        "total_trainable_M": round(
            (param_counts["mamba_trainable"] + param_counts["attn_trainable"]
             + param_counts["other_trainable"]) / 1e6, 3
        ),
        "pct_trainable":     param_counts["pct_trainable"],
        "best_eval_loss":    round(best_eval_loss, 4) if best_eval_loss else None,
        "best_eval_ppl":     round(best_eval_ppl,  2) if best_eval_ppl  else None,
        "grad_ratio_avg":    round(grad_stats.get("ratio_avg",   0.0), 3),
        "grad_ratio_final":  round(grad_stats.get("ratio_final", 0.0), 3),
        "mamba_grad_avg":    round(grad_stats.get("mamba_avg",   0.0), 4),
        "attn_grad_avg":     round(grad_stats.get("attn_avg",    0.0), 4),
    }
    path = out_dir / "metrics_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved metrics_summary.json → {path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_all_plots(cfg, trainer, monitor, model):
    out_dir = Path(cfg.output_dir) / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    param_counts = count_trainable_params(model)
    grad_stats   = monitor.report()
    magnitudes   = collect_adapter_magnitudes(model)

    plot_loss_curves(               trainer._train_loss_log, trainer._eval_loss_log, out_dir)
    plot_perplexity_curves(         trainer._train_loss_log, trainer._eval_loss_log, out_dir)
    plot_gradient_norm_ratio(       monitor,                                          out_dir)
    plot_param_breakdown(           param_counts, cfg,                                out_dir)
    plot_adapter_magnitude_heatmap( magnitudes,                                       out_dir)
    plot_lr_schedule(               trainer._lr_log,                                  out_dir)

    summary = save_metrics_json(cfg, param_counts, grad_stats, trainer.state, out_dir)

    logger.info("=" * 60)
    logger.info("Paper metrics summary:")
    for k, v in summary.items():
        logger.info(f"  {k:30s}: {v}")
    logger.info(f"All figures → {out_dir}/")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = ExperimentConfig()
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    model.config.use_cache = False

    # Uncomment to inspect module names before fine-tuning:
    # print_model_linear_modules(model)

    model   = apply_asymmetric_lora(model, cfg)
    monitor = GradientPathMonitor(model)

    train_data, eval_data = prepare_data(cfg, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=cfg.seed,
        gradient_checkpointing=True,
    )

    trainer = AsymmetricLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        monitor=monitor,
    )

    logger.info("Starting asymmetric LoRA fine-tuning …")
    trainer.train()
    monitor.remove()

    run_all_plots(cfg, trainer, monitor, model)

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info(f"Adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
