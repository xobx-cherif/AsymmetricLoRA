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
    mamba_targets: list[str],
    attn_targets: list[str],
    rank_m: int,
    rank_a: int,
    alpha_m: int,
    alpha_a: int,
    dropout: float = 0.05,
) -> tuple:
    """
    Return two separate LoraConfig objects — one per path — each with their
    own explicit rank and alpha.

    Why two configs instead of rank_pattern / alpha_pattern:
      PEFT's rank_pattern only overrides layers whose leaf name is an exact
      key in the dict.  Any unmatched leaf silently falls back to the base
      rank `r`.  This is the same class of silent failure that caused
      MOAFT's outproj_trainable_M=0.0 bug.  Two explicit configs guarantee
      the correct rank on every targeted layer with no silent fallback.

    For the symmetric case (rank_m == rank_a) both configs have equal rank;
    the two-pass injection is still safe and explicit.

    Returns:
        (mamba_cfg, attn_cfg) — pass to inject_lora_two_pass().
    """
    mamba_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank_m,
        lora_alpha=alpha_m,
        lora_dropout=dropout,
        target_modules=mamba_targets,
        bias="none",
    )
    attn_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank_a,
        lora_alpha=alpha_a,
        lora_dropout=dropout,
        target_modules=attn_targets,
        bias="none",
    )
    logger.info(
        "LoRA configs — Mamba: targets=%s r=%d alpha=%d | "
        "Attn: targets=%s r=%d alpha=%d",
        mamba_targets, rank_m, alpha_m,
        attn_targets, rank_a, alpha_a,
    )
    return mamba_cfg, attn_cfg


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


def inject_lora_two_pass(
    model: nn.Module,
    mamba_cfg: LoraConfig,
    attn_cfg: LoraConfig,
) -> nn.Module:
    """
    Inject LoRA adapters in two explicit passes — one per path.

    Pass 1: inject Mamba-path adapters (mamba.in_proj) at rank r_m.
    Pass 2: inject Attention-path adapters (q/k/v/o_proj) at rank r_a.

    After each pass, the actual trainable parameter counts are verified
    and logged.  An assertion guards against silent injection failure
    (the bug that caused MOAFT's outproj_trainable_M=0.0).

    Works for both symmetric (r_m == r_a) and asymmetric cases.
    """
    # Pass 1 — Mamba path
    model = get_peft_model(model, mamba_cfg)
    _verify_injection(model, mamba_cfg.target_modules, label="Mamba")

    # Pass 2 — Attention path
    # add_adapter injects a second adapter on top of the first without
    # removing the Mamba adapters already injected.
    model.add_adapter("attn_adapter", attn_cfg)
    model.set_adapter(["default", "attn_adapter"])
    _verify_injection(model, attn_cfg.target_modules, label="Attn")

    model.print_trainable_parameters()
    return model


def _verify_injection(
    model: nn.Module, targets: list[str], label: str
) -> None:
    """
    Assert that at least one trainable LoRA parameter exists for each
    target leaf name.  Raises AssertionError on silent injection failure.
    """
    trainable_names = {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    for leaf in targets:
        matched = [n for n in trainable_names if leaf in n and "lora_" in n]
        assert matched, (
            f"LoRA injection failed for {label} target '{leaf}': "
            f"no trainable lora_ parameter found. "
            f"Check target_modules and PEFT blacklist."
        )
    logger.info(
        "%s LoRA injection verified: all %d targets have trainable adapters.",
        label, len(targets),
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
