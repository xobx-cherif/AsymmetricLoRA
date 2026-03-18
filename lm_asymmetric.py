"""
lm_asymmetric.py — Path-Asymmetric LoRA fine-tuning of Falcon H1 on WikiText-2.

Assigns differentiated adapter ranks to the SSM and attention paths.
A single script covers both Strategy S2 (Mamba-high: r_m > r_a) and
Strategy S3 (Flipped: r_a > r_m) via command-line arguments.

Strategies from the paper:
  S2 (AsymLoRA):    --rank_m 16 --rank_a 4   (Mamba-high)
  S3 (Flipped):     --rank_m 4  --rank_a 16  (Attention-high)

Usage:
    # Strategy S2 — Mamba-high (paper default)
    python lm_asymmetric.py --rank_m 16 --rank_a 4

    # Strategy S3 — Flipped / Attention-high
    python lm_asymmetric.py --rank_m 4 --rank_a 16 \
        --output_dir ./outputs/lm_flipped

Outputs (in <output_dir>/figures/):
    loss_curves.pdf/png
    ppl_curve.pdf/png
    grad_ratio.pdf/png      rho over training — key diagnostic for rank direction
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
    inject_lora_two_pass,
    load_base_model,
    load_wikitext,
    logger,
    plot_loss_curves,
    plot_lr,
    plot_ppl_curve,
    plot_rho,
    save_metrics,
)

from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Path-asymmetric LoRA fine-tuning on WikiText-2"
    )
    p.add_argument("--model",        type=str,
                   default="tiiuae/Falcon-H1-1B-Base")
    # ── Core asymmetry parameters ────────────────────────────────────────────
    p.add_argument("--rank_m",       type=int, default=16,
                   help="LoRA rank for Mamba path (mamba.in_proj)")
    p.add_argument("--rank_a",       type=int, default=4,
                   help="LoRA rank for Attention path (q/k/v/o_proj)")
    p.add_argument("--alpha_m",      type=int, default=None,
                   help="LoRA alpha for Mamba path (default: 2 * rank_m)")
    p.add_argument("--alpha_a",      type=int, default=None,
                   help="LoRA alpha for Attention path (default: 2 * rank_a)")
    p.add_argument("--dropout",      type=float, default=0.05)
    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--num_train",    type=int, default=2000)
    p.add_argument("--num_eval",     type=int, default=200)
    p.add_argument("--max_length",   type=int, default=512)
    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--grad_accum",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_dir",   type=str,
                   default="./outputs/lm_asymmetric")
    p.add_argument("--logging_steps",type=int, default=20)
    p.add_argument("--eval_steps",   type=int, default=100)
    p.add_argument("--save_steps",   type=int, default=200)
    return p.parse_args()


def strategy_name(rank_m: int, rank_a: int) -> str:
    if rank_m > rank_a:
        return f"asym_lora_rm{rank_m}_ra{rank_a}"
    elif rank_a > rank_m:
        return f"flipped_lora_rm{rank_m}_ra{rank_a}"
    else:
        return f"symmetric_lora_r{rank_m}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    alpha_m = args.alpha_m if args.alpha_m is not None else 2 * args.rank_m
    alpha_a = args.alpha_a if args.alpha_a is not None else 2 * args.rank_a
    strat   = strategy_name(args.rank_m, args.rank_a)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"

    logger.info("=" * 60)
    logger.info("Path-Asymmetric LoRA — WikiText-2")
    logger.info("  strategy=%s", strat)
    logger.info("  rank_m=%d  alpha_m=%d  |  rank_a=%d  alpha_a=%d",
                args.rank_m, alpha_m, args.rank_a, alpha_a)
    logger.info("  model=%s", args.model)
    logger.info("=" * 60)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_base_model(args.model)
    mamba_tgts = get_mamba_targets(model)
    attn_tgts  = get_attention_targets(model)

    mamba_cfg, attn_cfg = build_lora_config(
        mamba_targets=mamba_tgts,
        attn_targets=attn_tgts,
        rank_m=args.rank_m,
        rank_a=args.rank_a,
        alpha_m=alpha_m,
        alpha_a=alpha_a,
        dropout=args.dropout,
    )
    model = inject_lora_two_pass(model, mamba_cfg, attn_cfg)
    param_counts = count_trainable_params(model)
    logger.info("Trainable params: %.3fM (%.2f%%)",
                param_counts["total_M"], param_counts["pct"])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, eval_ds = load_wikitext(
        tokenizer,
        num_train=args.num_train,
        num_eval=args.num_eval,
        max_length=args.max_length,
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
    rho_final  = round(float(rho_values[-1]), 4) if rho_values else None

    # Mamba and attn grad averages (from last logged step)
    mamba_grad_avg = None
    attn_grad_avg  = None
    if trainer.rho_log:
        # Re-compute from model's current state is not possible post-training;
        # rho_avg is the primary reported statistic.
        pass

    metrics = {
        "strategy":        strat,
        "model":           args.model,
        "rank_m":          args.rank_m,
        "rank_a":          args.rank_a,
        "alpha_m":         alpha_m,
        "alpha_a":         alpha_a,
        **param_counts,
        "best_eval_loss":  round(best_loss, 6) if best_loss else None,
        "best_eval_ppl":   round(best_ppl, 4)  if best_ppl  else None,
        "rho_avg":         rho_avg,
        "rho_final":       rho_final,
        "training_time":   train_time,
    }
    save_metrics(metrics, fig_dir)
    logger.info("Best PPL: %.4f | rho_avg: %.3f | Params: %.3fM",
                best_ppl or 0, rho_avg or 0, param_counts["total_M"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    title_base = f"{strat} (r_m={args.rank_m}, r_a={args.rank_a})"
    plot_loss_curves(trainer.train_loss_log, trainer.eval_loss_log,
                     f"{title_base} — WikiText-2 Loss", fig_dir)
    plot_ppl_curve(trainer.eval_loss_log,
                   f"{title_base} — WikiText-2 PPL", fig_dir)
    plot_rho(trainer.rho_log,
             f"ρ over training — {title_base}", fig_dir)
    plot_lr(trainer.lr_log, fig_dir)

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Adapter saved → %s", out_dir)


if __name__ == "__main__":
    main()
