"""
tooluse_eval.py — Standalone evaluation on the xLAM function-calling benchmark.

Evaluates either:
  (a) A fine-tuned checkpoint (PEFT adapter or full model save), or
  (b) The raw pretrained model with no fine-tuning (zero-shot baseline).

Metrics reported:
  FN-EM    — function name exact match (set comparison, order-independent)
  PK       — parameter key match (first call)
  PV       — parameter value match (first call)
  Full-EM  — all function names + call count + all param values correct
  Inv-JSON — fraction of outputs with no parseable JSON (lower = better)

The eval slice is reconstructed identically to the training split
(same shuffle seed and index offset) so that the zero-shot baseline
and fine-tuned models are evaluated on exactly the same examples.

Usage:
    # Evaluate a fine-tuned checkpoint
    python tooluse_eval.py --checkpoint ./outputs/tu_asymmetric

    # Zero-shot baseline (no fine-tuning)
    python tooluse_eval.py --zero_shot

    # Show first 5 examples in detail
    python tooluse_eval.py --checkpoint ./outputs/tu_flipped --verbose

    # After all strategies have been evaluated, generate comparison figure
    python tooluse_eval.py --compare --results_root ./outputs

Bug note: the original tool_use_finetuning.py extract_calls raised
IndexError because the regex fallback r"[.*]" has no capture group
but the code called .group(1). Fixed here by always using .group(0)
for uncaptured patterns and delegating to _try_parse_json.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from shared_utils import (
    XLAM_SYSTEM,
    XLAM_TEMPLATE,
    STRATEGY_COLORS,
    SYM_CLR, MAMBA_CLR, FLIP_CLR,
    logger,
    save_metrics,
    savefig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt for evaluation (prompt-only, no answer)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PROMPT_TEMPLATE = (
    "<s>\n{system}\n</s>\n\n"
    "<tools>\n{tools}\n</tools>\n\n"
    "<user>\n{query}\n</user>\n\n"
    "<calls>\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed JSON extractor
# ─────────────────────────────────────────────────────────────────────────────

def _try_parse_json(raw: str) -> list:
    """
    Attempt JSON parsing with progressive cleanup.
    Returns a list of dicts, or [] on failure.
    """
    if not raw:
        return []
    for attempt in (
        raw,
        raw.rstrip(",").strip(),
        re.sub(r"[^}\]]+$", "", raw).strip(),
    ):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
        except (json.JSONDecodeError, ValueError):
            continue
    return []


def extract_calls(text: str) -> list:
    """
    Extract JSON function calls from model output using four fallback strategies.
    All uncaptured regex patterns use .group(0) to avoid IndexError.

    Strategy 1: <calls>...</calls> — uses group(1) (captured)
    Strategy 2: <calls> prefix    — uses group(1) (captured)
    Strategy 3: bare JSON array   — uses group(0) (not captured)
    Strategy 4: bare JSON object  — uses group(0) (not captured)
    """
    m = re.search(r"<calls>(.*?)</calls>", text, re.DOTALL)
    if m:
        result = _try_parse_json(m.group(1).strip())
        if result:
            return result

    m = re.search(r"<calls>\s*([\s\S]+)", text)
    if m:
        result = _try_parse_json(m.group(1).strip())
        if result:
            return result

    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        result = _try_parse_json(m.group(0).strip())   # group(0) — no capture group
        if result:
            return result

    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        result = _try_parse_json(m.group(0).strip())   # group(0) — no capture group
        if result:
            return result

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _get_fn_name(call: dict) -> str:
    return call.get("name", call.get("function", call.get("tool", "")))


def _get_args(call: dict) -> dict:
    args = call.get("arguments",
           call.get("parameters",
           call.get("args",
           call.get("kwargs", {}))))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def score_call(pred_calls: list, gold_calls: list) -> dict:
    """
    Score one example. Returns a dict of per-metric 0/1 values.

    fn_exact_match    — set equality of function names (order-independent)
    param_key_match   — key set equality for the first call
    param_value_match — key+value equality for the first call
    full_exact_match  — fn_em AND param_value_match AND correct call count
    invalid_json      — 1 if no parseable JSON was produced
    num_calls_correct — 1 if predicted call count matches gold
    """
    if not pred_calls:
        return dict(fn_exact_match=0, param_key_match=0, param_value_match=0,
                    full_exact_match=0, invalid_json=1, num_calls_correct=0)
    try:
        gold_fns = {_get_fn_name(c) for c in gold_calls}
        pred_fns = {_get_fn_name(c) for c in pred_calls}
        fn_em  = int(gold_fns == pred_fns)
        n_ok   = int(len(pred_calls) == len(gold_calls))
        g_args = _get_args(gold_calls[0])
        p_args = _get_args(pred_calls[0])
        key_ok = int(set(g_args.keys()) == set(p_args.keys()))
        val_ok = int(g_args == p_args)
        full   = int(fn_em and val_ok and n_ok)
        return dict(fn_exact_match=fn_em, param_key_match=key_ok,
                    param_value_match=val_ok, full_exact_match=full,
                    invalid_json=0, num_calls_correct=n_ok)
    except Exception as e:
        logger.debug("Scoring error: %s", e)
        return dict(fn_exact_match=0, param_key_match=0, param_value_match=0,
                    full_exact_match=0, invalid_json=1, num_calls_correct=0)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    checkpoint: Optional[str] = None,
    zero_shot: bool = False,
):
    """
    Load tokenizer and model.
    If zero_shot=True, loads the base model with no adapters.
    If checkpoint is provided, detects PEFT vs full model save automatically.
    """
    source = checkpoint if checkpoint else model_name
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if zero_shot:
        logger.info("Loading base model (zero-shot): %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
    elif checkpoint and (Path(checkpoint) / "adapter_config.json").exists():
        logger.info("Loading PEFT adapter from %s on top of %s",
                    checkpoint, model_name)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        logger.info("Loading full model from %s", checkpoint or model_name)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint or model_name, quantization_config=bnb,
            torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.1fM total, %.1fM trainable",
                total / 1e6, trainable / 1e6)
    if zero_shot and trainable > 0:
        logger.warning("trainable > 0 in zero-shot mode — verify no adapters loaded")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_slice(num_eval: int, num_train: int = 2000, seed: int = 42):
    """
    Reconstruct the same held-out eval slice used during training.
    num_train must match the value passed to tooluse_finetune.py (default 2000).
    """
    logger.info("Loading xLAM eval slice (seed=%d, skip=%d, take=%d) …",
                seed, num_train, num_eval)
    raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    raw = raw.shuffle(seed=seed)
    raw = raw.select(range(min(num_train + num_eval, len(raw))))
    eval_raw = raw.select(range(num_train, num_train + num_eval))
    logger.info("Eval slice: %d examples (indices %d–%d)",
                len(eval_raw), num_train, num_train + len(eval_raw) - 1)
    return eval_raw


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model,
    tokenizer,
    eval_raw,
    max_length: int = 1024,
    max_new_tokens: int = 256,
    num_samples: int = 200,
    verbose: bool = False,
) -> dict:
    device = next(model.parameters()).device
    n = min(num_samples, len(eval_raw))
    logger.info("Evaluating %d examples on device %s …", n, device)

    all_scores: dict[str, list] = {}

    for i in range(n):
        row      = eval_raw[i]
        tools_str = (row["tools"] if isinstance(row["tools"], str)
                     else json.dumps(row["tools"], indent=2))
        gold_str  = (row["answers"] if isinstance(row["answers"], str)
                     else json.dumps(row["answers"]))

        prompt = EVAL_PROMPT_TEMPLATE.format(
            system=XLAM_SYSTEM,
            tools=tools_str,
            query=row["query"],
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_length - max_new_tokens,
        ).to(device)

        try:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        except Exception as e:
            logger.warning("Generation error on example %d: %s", i, e)
            generated = ""

        try:
            gold_calls = json.loads(gold_str)
            if isinstance(gold_calls, dict):
                gold_calls = [gold_calls]
        except json.JSONDecodeError:
            continue

        pred_calls = extract_calls(generated)
        scores     = score_call(pred_calls, gold_calls)

        for k, v in scores.items():
            all_scores.setdefault(k, []).append(v)

        if verbose or i < 5:
            logger.info("--- Example %d ---", i)
            logger.info("Query:     %s", row["query"][:80])
            logger.info("Gold:      %s", gold_str[:120])
            logger.info("Generated: %s", generated[:200])
            logger.info("Parsed:    %s", pred_calls)
            logger.info("Scores:    %s", scores)

        if (i + 1) % 50 == 0:
            interim = {k: round(float(np.mean(v)), 3)
                       for k, v in all_scores.items()}
            logger.info("[%d/%d] running: %s", i + 1, n, interim)

    metrics = {k: round(float(np.mean(v)), 4)
               for k, v in all_scores.items() if v}

    logger.info("=" * 55)
    logger.info("EVALUATION RESULTS:")
    for k, v in metrics.items():
        logger.info("  %-28s: %.4f", k, v)
    logger.info("=" * 55)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_eval_results(
    metrics: dict, out_dir: Path, strategy_name: str, model_name: str,
) -> None:
    fig_dir = out_dir / "figures"

    # Merge into existing metrics.json if present
    existing_path = fig_dir / "metrics.json"
    summary = {}
    if existing_path.exists():
        with open(existing_path) as f:
            summary = json.load(f)
    summary.update({f"tool_{k}": v for k, v in metrics.items()})
    summary["strategy"]  = strategy_name
    summary["model"]     = model_name
    save_metrics(summary, fig_dir)

    # Bar chart
    labels_map = {
        "fn_exact_match":    "FN-EM",
        "param_key_match":   "PK",
        "param_value_match": "PV",
        "full_exact_match":  "Full-EM",
        "invalid_json":      "Inv-JSON ↓",
    }
    keys   = [k for k in labels_map if k in metrics]
    labels = [labels_map[k] for k in keys]
    values = [metrics[k] for k in keys]
    colors = ["#d62728" if k == "invalid_json" else MAMBA_CLR for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Tool-use metrics — {strategy_name}")
    fig.tight_layout()
    savefig(fig, fig_dir, "tool_eval_bar")


def plot_comparison(results_root: str) -> None:
    """Load all strategy metrics and produce a cross-strategy comparison figure."""
    root       = Path(results_root)
    strategies = [
        ("zero_shot",  "./outputs/baseline_no_ft"),
        ("symmetric",  "./outputs/tu_symmetric"),
        ("asymmetric", "./outputs/tu_asymmetric"),
        ("flipped",    "./outputs/tu_flipped"),
    ]
    all_data: dict[str, dict] = {}
    for name, default_path in strategies:
        for p in (root / default_path, root / name):
            mpath = p / "figures" / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    data = json.load(f)
                tm = {k.replace("tool_", ""): v for k, v in data.items()
                      if k.startswith("tool_")}
                if tm:
                    all_data[name] = tm
                    logger.info("Loaded %s: %s", name, tm)
                break
        else:
            logger.warning("No metrics found for strategy: %s", name)

    if len(all_data) < 2:
        logger.warning("Need ≥ 2 strategies to compare. Run eval first.")
        return

    metrics_to_plot = ["full_exact_match", "fn_exact_match",
                       "param_value_match", "invalid_json"]
    titles = ["Full-EM ↑", "FN-EM ↑", "PV ↑", "Inv-JSON ↓"]
    clr_map = {
        "zero_shot":  SYM_CLR,
        "symmetric":  SYM_CLR,
        "asymmetric": MAMBA_CLR,
        "flipped":    FLIP_CLR,
    }

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        names = [s for s in all_data if metric in all_data[s]]
        vals  = [all_data[s][metric] for s in names]
        cols  = [clr_map.get(s, "#888") for s in names]
        lbls  = [s.replace("_", "\n") for s in names]
        bars  = ax.bar(lbls, vals, color=cols, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Tool-use performance across fine-tuning strategies — Falcon H1-1B\n"
        "xLAM function-calling benchmark (200 held-out examples, greedy decoding)",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout()
    cmp_dir = Path(results_root) / "comparison"
    savefig(fig, cmp_dir, "tool_use_comparison")
    logger.info("Comparison figure → %s", cmp_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone tool-use evaluation for Falcon H1"
    )
    p.add_argument("--checkpoint",     type=str, default=None,
                   help="Path to fine-tuned checkpoint directory")
    p.add_argument("--zero_shot",      action="store_true",
                   help="Evaluate the raw pretrained model (no adapters)")
    p.add_argument("--model",          type=str,
                   default="tiiuae/Falcon-H1-1B-Base",
                   help="Base model name (used for zero-shot or adapter loading)")
    p.add_argument("--strategy",       type=str, default=None,
                   help="Strategy label for output files "
                        "(inferred from checkpoint dir if omitted)")
    p.add_argument("--num_eval",       type=int, default=200)
    p.add_argument("--num_train",      type=int, default=2000,
                   help="Must match num_train used in tooluse_finetune.py")
    p.add_argument("--max_length",     type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--verbose",        action="store_true",
                   help="Print first 5 examples in full")
    p.add_argument("--compare",        action="store_true",
                   help="Generate cross-strategy comparison from saved results")
    p.add_argument("--results_root",   type=str, default="./outputs",
                   help="Root dir for --compare mode")
    p.add_argument("--output_dir",     type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        plot_comparison(args.results_root)
        return

    if not args.checkpoint and not args.zero_shot:
        raise ValueError(
            "Provide --checkpoint <dir> for a fine-tuned model, "
            "or --zero_shot for the baseline."
        )

    strategy_name = (args.strategy
                     or ("zero_shot" if args.zero_shot
                         else Path(args.checkpoint).name))
    out_dir = Path(args.output_dir
                   or (args.checkpoint if not args.zero_shot
                       else "./outputs/baseline_no_ft"))

    logger.info("Strategy: %s", strategy_name)
    if args.zero_shot:
        logger.info("Mode: zero-shot (no fine-tuning)")
    else:
        logger.info("Checkpoint: %s", args.checkpoint)

    model, tokenizer = load_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        zero_shot=args.zero_shot,
    )
    eval_raw = load_eval_slice(args.num_eval, args.num_train, args.seed)

    metrics = run_evaluation(
        model, tokenizer, eval_raw,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_eval,
        verbose=args.verbose,
    )

    save_eval_results(metrics, out_dir, strategy_name, args.model)


if __name__ == "__main__":
    main()
