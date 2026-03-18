"""
shared_utils.py — Common utilities for Falcon H1 fine-tuning experiments.

Provides:
  - PEFT helpers: safe target-module detection, blacklist enforcement
  - Data helpers: WikiText-2 and xLAM dataset loaders
  - Metric helpers: perplexity, trainable param counting, gradient norm ratio
  - Plot helpers: consistent figure saving and style
  - Trainer subclass: logs loss/LR history for downstream plotting
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("falcon_h1")

# ── plot style ────────────────────────────────────────────────────────────────
MAMBA_CLR = "#1f77b4"
ATTN_CLR  = "#9467bd"
FLIP_CLR  = "#d65f2b"
SYM_CLR   = "#505050"
TRAIN_CLR = "#2ca02c"
EVAL_CLR  = "#d62728"

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

STRATEGY_COLORS = {
    "symmetric":      SYM_CLR,
    "asymmetric":     MAMBA_CLR,
    "flipped":        FLIP_CLR,
}


# ── PEFT helpers ──────────────────────────────────────────────────────────────

# PEFT blocks these leaf names for model_type=falcon_h1.
# Attempting to include them raises ValueError.
PEFT_BLACKLIST = frozenset({"out_proj", "conv1d"})


def get_mamba_targets(model: nn.Module) -> list[str]:
    """
    Return PEFT-safe leaf names for the Mamba path of Falcon H1.
    Confirmed safe: in_proj (2048→6704).
    Blocked by PEFT: out_proj (3072→2048).
    """
    found: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        is_mamba = "mamba" in lower or any(
            t in lower for t in ("in_proj", "x_proj", "dt_proj")
        )
        is_attn = "attn" in lower or "attention" in lower
        if is_mamba and not is_attn:
            leaf = name.split(".")[-1]
            if leaf not in PEFT_BLACKLIST and leaf not in found:
                found.append(leaf)
    if not found:
        logger.warning("No Mamba targets detected — falling back to ['in_proj']")
        found = ["in_proj"]
    logger.info("Mamba LoRA targets: %s", found)
    return found


def get_attention_targets(model: nn.Module) -> list[str]:
    """Return PEFT-safe leaf names for the attention path of Falcon H1."""
    found: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        if "attn" in lower or "attention" in lower:
            leaf = name.split(".")[-1]
            if leaf not in PEFT_BLACKLIST and leaf not in found:
                found.append(leaf)
    if not found:
        logger.warning("No Attention targets detected — falling back to "
                       "['q_proj','k_proj','v_proj','o_proj']")
        found = ["q_proj", "k_proj", "v_proj", "o_proj"]
    logger.info("Attention LoRA targets: %s", found)
    return found


def build_lora_config(
    model: nn.Module,
    mamba_targets: list[str],
    attn_targets: list[str],
    rank_m: int,
    rank_a: int,
    alpha_m: int,
    alpha_a: int,
    dropout: float = 0.05,
) -> LoraConfig:
    """
    Build a single LoraConfig with correct per-path rank assignment.

    IMPORTANT — rank_pattern key semantics:
      PEFT resolves rank via:
          r = config.rank_pattern.get(key, config.r)
      where `key` is the FULL dotted module name from named_modules()
      (e.g. "layers.0.self_attn.q_proj"), NOT just the leaf name.
      Using only leaf names ("q_proj") as keys causes silent fallback
      to base_r for every layer — the asymmetry is never applied.

    This function walks model.named_modules() to build rank_pattern and
    alpha_pattern with full dotted paths as keys, guaranteeing the correct
    rank on every layer with no silent fallback.

    The `model` argument is required so we can enumerate the actual
    module paths.
    """
    all_targets = list(dict.fromkeys(mamba_targets + attn_targets))
    rank_pattern: dict[str, int]  = {}
    alpha_pattern: dict[str, int] = {}

    mamba_set = set(mamba_targets)
    attn_set  = set(attn_targets)

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split(".")[-1]
        if leaf in mamba_set:
            rank_pattern[full_name]  = rank_m
            alpha_pattern[full_name] = alpha_m
        elif leaf in attn_set:
            rank_pattern[full_name]  = rank_a
            alpha_pattern[full_name] = alpha_a

    n_mamba = sum(1 for v in rank_pattern.values() if v == rank_m)
    n_attn  = sum(1 for v in rank_pattern.values() if v == rank_a)
    logger.info(
        "rank_pattern built with full paths: "
        "%d Mamba layers (r=%d, alpha=%d), "
        "%d Attn layers (r=%d, alpha=%d)",
        n_mamba, rank_m, alpha_m,
        n_attn,  rank_a, alpha_a,
    )

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank_m,               # base fallback — should never be used
        lora_alpha=alpha_m,     # base fallback — should never be used
        lora_dropout=dropout,
        target_modules=all_targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )


def load_base_model(model_name: str) -> tuple:
    """
    Load Falcon H1 in 4-bit NF4 QLoRA configuration.
    Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    logger.info("Loaded %s", model_name)
    return model, tokenizer


def inject_lora(model: nn.Module, cfg: LoraConfig) -> nn.Module:
    """
    Wrap model with PEFT LoRA adapters using a single LoraConfig.

    A single get_peft_model call is correct for training: all adapter
    parameters — both Mamba-path and attention-path — are injected into
    one unified model wrapper. AdamW receives a single parameter list and
    updates all adapters in the same backward pass.

    After injection, verify_lora_ranks() asserts that every target layer
    received the intended rank (guards against silent rank_pattern fallback).
    """
    model = get_peft_model(model, cfg)
    verify_lora_ranks(model, cfg)
    model.print_trainable_parameters()
    return model


def verify_lora_ranks(model: nn.Module, cfg: LoraConfig) -> None:
    """
    Assert that every LoRA adapter in the model has the rank specified
    by cfg.rank_pattern (or cfg.r if the leaf is not in rank_pattern).

    Guards against the silent fallback behaviour of rank_pattern: if a
    leaf name is not found in rank_pattern, PEFT uses the base cfg.r.
    This check makes that fallback visible as an AssertionError rather
    than a silent correctness bug.

    Also asserts that at least one lora_ parameter exists for each entry
    in cfg.target_modules, catching complete injection failures.
    """
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # 1. Every target leaf must have at least one lora_ parameter
    for leaf in cfg.target_modules:
        matched = [n for n in trainable if leaf in n and "lora_" in n]
        assert matched, (
            f"LoRA injection failed for target '{leaf}': "
            f"no trainable lora_ parameter found. "
            f"Check target_modules and PEFT blacklist."
        )

    # 2. Verify each lora_A matrix has the expected rank (shape[0] == r)
    rank_pattern = getattr(cfg, "rank_pattern", {}) or {}
    for name, param in trainable.items():
        if "lora_A" not in name:
            continue
        leaf = name.split(".")[-3] if "lora_A" in name else None
        if leaf is None:
            continue
        expected_r = rank_pattern.get(leaf, cfg.r)
        actual_r   = param.shape[0]
        assert actual_r == expected_r, (
            f"Rank mismatch for {name}: "
            f"expected r={expected_r} (from rank_pattern[{leaf!r}]), "
            f"got r={actual_r}. "
            f"Ensure '{leaf}' is in rank_pattern with the correct value."
        )

    logger.info(
        "verify_lora_ranks: all %d lora_A matrices have correct ranks.",
        sum(1 for n in trainable if "lora_A" in n),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 30.0))


def count_trainable_params(model: nn.Module) -> dict:
    """
    Count trainable parameters split by sub-module type.
    Returns a dict with mamba_M, attn_M, total_M, pct keys.
    """
    mamba_p = attn_p = other_p = frozen_p = 0
    for name, param in model.named_parameters():
        lower = name.lower()
        n = param.numel()
        if not param.requires_grad:
            frozen_p += n
        elif "mamba" in lower or any(t in lower for t in ("in_proj", "x_proj", "dt_proj")):
            mamba_p += n
        elif "attn" in lower or "attention" in lower:
            attn_p += n
        else:
            other_p += n
    total_all = mamba_p + attn_p + other_p + frozen_p
    trainable  = mamba_p + attn_p + other_p
    return {
        "mamba_M":   round(mamba_p / 1e6, 3),
        "attn_M":    round(attn_p  / 1e6, 3),
        "other_M":   round(other_p / 1e6, 3),
        "total_M":   round(trainable / 1e6, 3),
        "pct":       round(100.0 * trainable / max(total_all, 1), 4),
    }


def compute_grad_ratio(model: nn.Module, eps: float = 1e-9) -> float:
    """
    Compute rho = ||grad_mamba||_F / (||grad_attn||_F + eps)
    over the current gradient buffers.
    Returns 0.0 if no gradients are available.
    """
    mamba_sq = attn_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        lower = name.lower()
        g2 = param.grad.float().norm().item() ** 2
        if "mamba" in lower or "in_proj" in lower:
            mamba_sq += g2
        elif "attn" in lower or "attention" in lower:
            attn_sq += g2
    return math.sqrt(mamba_sq) / (math.sqrt(attn_sq) + eps)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_wikitext(
    tokenizer,
    num_train: int = 2000,
    num_eval: int = 200,
    max_length: int = 512,
) -> tuple:
    """
    Load and tokenise WikiText-2.
    Returns (train_dataset, eval_dataset).
    """
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_raw = raw["train"].select(range(num_train))
    eval_raw  = raw["validation"].select(range(num_eval))

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_ds = train_raw.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds  = eval_raw.map( tokenize, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    logger.info("WikiText-2: %d train, %d eval", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


# xLAM prompt template (must match eval exactly)
XLAM_SYSTEM = (
    "You are an AI assistant that calls functions when needed. "
    "Given a user query and available tools, output the correct function call "
    "in JSON format inside <calls> tags. "
    "If no function call is needed, respond normally."
)

XLAM_TEMPLATE = (
    "<s>\n{system}\n</s>\n\n"
    "<tools>\n{tools}\n</tools>\n\n"
    "<user>\n{query}\n</user>\n\n"
    "<calls>\n{answer}</calls>"
)


def load_xlam(
    tokenizer,
    num_train: int = 2000,
    num_eval: int = 200,
    max_length: int = 1024,
    seed: int = 42,
) -> tuple:
    """
    Load, shuffle, split, and tokenise the xLAM function-calling dataset.
    Loss is computed only on the <calls>...</calls> completion tokens;
    the system prompt, tool schemas, and user query are masked with -100.

    Returns (train_dataset, eval_dataset, eval_raw).
    eval_raw is the untokenised eval slice needed for generation-based eval.
    """
    logger.info("Loading Salesforce/xlam-function-calling-60k …")
    raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    raw = raw.shuffle(seed=seed)
    total = num_train + num_eval
    raw   = raw.select(range(min(total, len(raw))))
    train_raw = raw.select(range(num_train))
    eval_raw  = raw.select(range(num_train, num_train + num_eval))

    def process(examples):
        ids_list, mask_list, label_list = [], [], []
        for i in range(len(examples["query"])):
            tools_str = (examples["tools"][i]
                         if isinstance(examples["tools"][i], str)
                         else json.dumps(examples["tools"][i], indent=2))
            ans_str   = (examples["answers"][i]
                         if isinstance(examples["answers"][i], str)
                         else json.dumps(examples["answers"][i]))

            full_text = XLAM_TEMPLATE.format(
                system=XLAM_SYSTEM,
                tools=tools_str,
                query=examples["query"][i],
                answer=ans_str,
            )
            enc = tokenizer(
                full_text, truncation=True,
                max_length=max_length, padding=False,
            )
            input_ids = enc["input_ids"]

            # Mask everything before the answer
            prompt_text = XLAM_TEMPLATE.format(
                system=XLAM_SYSTEM,
                tools=tools_str,
                query=examples["query"][i],
                answer="",
            )
            prompt_len = len(tokenizer(
                prompt_text, truncation=True,
                max_length=max_length, padding=False,
            )["input_ids"])

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:len(input_ids)]

            ids_list.append(input_ids)
            mask_list.append([1] * len(input_ids))
            label_list.append(labels)

        return {
            "input_ids":      ids_list,
            "attention_mask": mask_list,
            "labels":         label_list,
        }

    train_ds = train_raw.map(
        process, batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenising train",
    )
    eval_ds = eval_raw.map(
        process, batched=True,
        remove_columns=eval_raw.column_names,
        desc="Tokenising eval",
    )
    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    logger.info("xLAM: %d train, %d eval", len(train_ds), len(eval_ds))
    return train_ds, eval_ds, eval_raw


# ── Custom Trainer ────────────────────────────────────────────────────────────

class LoggingTrainer(Trainer):
    """
    Trainer subclass that records train loss, eval loss, and LR
    at every logging step for downstream plotting.
    Also logs the gradient norm ratio rho at each step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loss_log: list[tuple[int, float]] = []
        self.eval_loss_log:  list[tuple[int, float]] = []
        self.lr_log:         list[tuple[int, float]] = []
        self.rho_log:        list[tuple[int, float]] = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        step = self.state.global_step
        if step % self.args.logging_steps == 0:
            rho = compute_grad_ratio(model)
            self.rho_log.append((step, rho))
        return loss

    def log(self, logs: dict, start_time: Optional[float] = None):
        super().log(logs)
        step = self.state.global_step
        if "loss" in logs:
            self.train_loss_log.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_loss_log.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.lr_log.append((step, logs["learning_rate"]))


# ── Plot helpers ──────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    train_log: list, eval_log: list,
    title: str, out_dir: Path, stem: str = "loss_curves",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if train_log:
        s, v = zip(*train_log)
        ax.plot(s, v, color=TRAIN_CLR, lw=1.8, label="Train loss")
    if eval_log:
        s, v = zip(*eval_log)
        ax.plot(s, v, color=EVAL_CLR, lw=1.8, ls="--",
                marker="o", ms=4, label="Eval loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title); ax.legend()
    savefig(fig, out_dir, stem)


def plot_ppl_curve(
    eval_log: list, title: str, out_dir: Path, stem: str = "ppl_curve",
) -> None:
    if not eval_log:
        return
    s, v = zip(*eval_log)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, [compute_perplexity(x) for x in v],
            color=EVAL_CLR, lw=1.8, marker="o", ms=4)
    ax.set_xlabel("Step"); ax.set_ylabel("Perplexity")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    savefig(fig, out_dir, stem)


def plot_rho(
    rho_log: list, title: str, out_dir: Path, stem: str = "grad_ratio",
) -> None:
    if not rho_log:
        return
    s, v = zip(*rho_log)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, v, color=MAMBA_CLR, lw=1.8,
            label=r"$\rho$ (Mamba / Attn grad norm ratio)")
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, label=r"$\rho = 1$")
    ax.set_xlabel("Step"); ax.set_ylabel(r"$\rho$")
    ax.set_title(title); ax.legend()
    savefig(fig, out_dir, stem)


def plot_lr(
    lr_log: list, out_dir: Path, stem: str = "lr_schedule",
) -> None:
    if not lr_log:
        return
    s, v = zip(*lr_log)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(s, v, color="#17becf", lw=1.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule (cosine with warmup)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    savefig(fig, out_dir, stem)


# ── Metrics JSON ──────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, out_dir: Path, filename: str = "metrics.json") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", path)
