"""
tooluse_finetune.py — Fine-tuning Falcon H1 on xLAM function-calling.

Covers all three LoRA strategies via --strategy flag:
  symmetric   : uniform rank r=16 (S1)
  asymmetric  : rank_m=16, rank_a=4  (S2 — Mamba-high)
  flipped     : rank_m=4,  rank_a=16 (S3 — Attention-high)

Loss is computed only on the <calls>...</calls> completion tokens.
System prompt, tool schemas, and user query are masked with label=-100.

Usage:
    python tooluse_finetune.py --strategy asymmetric
    python tooluse_finetune.py --strategy flipped --output_dir ./outputs/tu_flipped
    python tooluse_finetune.py --strategy symmetric

Outputs (in <output_dir>/figures/):
    loss_curves.pdf/png
    ppl_curve.pdf/png
    grad_ratio.pdf/png
    lr_schedule.pdf/png
    metrics.json
"""

import argparse
import os
import time
from pathlib import Path

import torch

from shared_utils import (
    LoggingTrainer,
    build_lora_config,
    compute_perplexity,
    count_trainable_params,
    get_attention_targets,
    get_mamba_targets,
    inject_lora,
    load_base_model,
    load_xlam,
    logger,
    plot_loss_curves,
    plot_lr,
    plot_ppl_curve,
    plot_rho,
    save_metrics,
)

from transformers import (
    DataCollatorForSeq2Seq,
    TrainingArguments,
)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy presets
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "symmetric": {
        "rank_m": 16, "rank_a": 16,
        "alpha_m": 32, "alpha_a": 32,
    },
    "asymmetric": {
        "rank_m": 16, "rank_a": 4,
        "alpha_m": 32, "alpha_a": 8,
    },
    "flipped": {
        "rank_m": 4, "rank_a": 16,
        "alpha_m": 8, "alpha_a": 32,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tool-use fine-tuning on xLAM function-calling"
    )
    p.add_argument("--strategy",     type=str, default="asymmetric",
                   choices=list(STRATEGIES.keys()),
                   help="LoRA rank strategy")
    p.add_argument("--model",        type=str,
                   default="tiiuae/Falcon-H1-1B-Base")
    # Override rank/alpha (optional; defaults come from STRATEGIES)
    p.add_argument("--rank_m",       type=int, default=None,
                   help="Override Mamba rank from strategy preset")
    p.add_argument("--rank_a",       type=int, default=None,
                   help="Override Attention rank from strategy preset")
    p.add_argument("--dropout",      type=float, default=0.05)
    # Data
    p.add_argument("--num_train",    type=int, default=2000)
    p.add_argument("--num_eval",     type=int, default=200)
    p.add_argument("--max_length",   type=int, default=1024)
    p.add_argument("--seed",         type=int, default=42)
    # Training
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--batch_size",   type=int, default=2)
    p.add_argument("--grad_accum",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--output_dir",   type=str, default=None,
                   help="Default: ./outputs/tu_<strategy>")
    p.add_argument("--logging_steps",type=int, default=20)
    p.add_argument("--eval_steps",   type=int, default=100)
    p.add_argument("--save_steps",   type=int, default=200)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    preset = STRATEGIES[args.strategy].copy()
    if args.rank_m is not None:
        preset["rank_m"]  = args.rank_m
        preset["alpha_m"] = 2 * args.rank_m
    if args.rank_a is not None:
        preset["rank_a"]  = args.rank_a
        preset["alpha_a"] = 2 * args.rank_a

    out_dir = Path(args.output_dir or f"./outputs/tu_{args.strategy}")
    fig_dir = out_dir / "figures"
    os.makedirs(str(out_dir), exist_ok=True)
    torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Tool-use fine-tuning — xLAM")
    logger.info("  strategy=%s", args.strategy)
    logger.info("  rank_m=%d  alpha_m=%d  |  rank_a=%d  alpha_a=%d",
                preset["rank_m"], preset["alpha_m"],
                preset["rank_a"], preset["alpha_a"])
    logger.info("  model=%s", args.model)
    logger.info("=" * 60)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_base_model(args.model)
    mamba_tgts = get_mamba_targets(model)
    attn_tgts  = get_attention_targets(model)

    lora_cfg = build_lora_config(
        mamba_targets=mamba_tgts,
        attn_targets=attn_tgts,
        rank_m=preset["rank_m"],
        rank_a=preset["rank_a"],
        alpha_m=preset["alpha_m"],
        alpha_a=preset["alpha_a"],
        dropout=args.dropout,
    )
    model = inject_lora(model, lora_cfg)
    param_counts = count_trainable_params(model)
    logger.info("Trainable params: %.3fM (%.2f%%)",
                param_counts["total_M"], param_counts["pct"])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, eval_ds, _ = load_xlam(
        tokenizer,
        num_train=args.num_train,
        num_eval=args.num_eval,
        max_length=args.max_length,
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model,
        padding=True, pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed   = time.time() - t0
    h, m, s   = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    train_time = f"{h}:{m:02d}:{s:02d}"
    logger.info("Training complete in %s", train_time)

    # ── Results ───────────────────────────────────────────────────────────────
    evals = [e for e in trainer.state.log_history if "eval_loss" in e]
    best_loss = min(e["eval_loss"] for e in evals) if evals else None
    best_ppl  = compute_perplexity(best_loss) if best_loss else None

    rho_values = [v for _, v in trainer.rho_log]
    rho_avg    = round(float(sum(rho_values) / len(rho_values)), 4) if rho_values else None

    metrics = {
        "strategy":        args.strategy,
        "model":           args.model,
        **preset,
        **param_counts,
        "best_eval_loss":  round(best_loss, 6) if best_loss else None,
        "best_eval_ppl":   round(best_ppl, 4)  if best_ppl  else None,
        "rho_avg":         rho_avg,
        "training_time":   train_time,
    }
    save_metrics(metrics, fig_dir)
    logger.info("Best PPL: %.4f | Params: %.3fM", best_ppl or 0, param_counts["total_M"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    title = f"Tool-use {args.strategy} (r_m={preset['rank_m']}, r_a={preset['rank_a']})"
    plot_loss_curves(trainer.train_loss_log, trainer.eval_loss_log,
                     f"{title} — Loss", fig_dir)
    plot_ppl_curve(trainer.eval_loss_log, f"{title} — PPL", fig_dir)
    plot_rho(trainer.rho_log, f"ρ over training — {title}", fig_dir)
    plot_lr(trainer.lr_log, fig_dir)

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Adapter saved → %s", out_dir)


if __name__ == "__main__":
    main()
