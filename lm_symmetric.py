"""
lm_symmetric.py — Symmetric LoRA fine-tuning of Falcon H1 on WikiText-2.

Applies a uniform LoRA rank to all eligible linear layers (mamba.in_proj
and attention q/k/v/o_proj). Serves as the parameter-efficiency baseline
for the asymmetric experiments.

Strategy S1 from the paper:
  Delta_W = (alpha_s / r_s) * B * A  for all W in Theta

Usage:
    python lm_symmetric.py
    python lm_symmetric.py --rank 16 --output_dir ./outputs/lm_symmetric_r16

Outputs (in <output_dir>/figures/):
    loss_curves.pdf/png     train + eval loss over steps
    ppl_curve.pdf/png       eval perplexity over steps
    grad_ratio.pdf/png      gradient norm ratio rho over steps
    lr_schedule.pdf/png     learning rate schedule
    metrics.json            all scalar results (for LaTeX table)
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
    load_wikitext,
    logger,
    plot_loss_curves,
    plot_lr,
    plot_ppl_curve,
    plot_rho,
    save_metrics,
    savefig,
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
        description="Symmetric LoRA fine-tuning on WikiText-2"
    )
    p.add_argument("--model",        type=str,
                   default="tiiuae/Falcon-H1-1B-Base")
    p.add_argument("--rank",         type=int, default=16,
                   help="Uniform LoRA rank applied to all target layers")
    p.add_argument("--alpha",        type=int, default=None,
                   help="LoRA alpha (default: 2 * rank)")
    p.add_argument("--dropout",      type=float, default=0.05)
    p.add_argument("--num_train",    type=int, default=2000)
    p.add_argument("--num_eval",     type=int, default=200)
    p.add_argument("--max_length",   type=int, default=512)
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--grad_accum",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_dir",   type=str,
                   default="./outputs/lm_symmetric")
    p.add_argument("--logging_steps",type=int, default=20)
    p.add_argument("--eval_steps",   type=int, default=100)
    p.add_argument("--save_steps",   type=int, default=200)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"

    logger.info("=" * 60)
    logger.info("Symmetric LoRA — WikiText-2")
    logger.info("  rank=%d  alpha=%d  model=%s", args.rank, alpha, args.model)
    logger.info("=" * 60)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_base_model(args.model)
    mamba_tgts = get_mamba_targets(model)
    attn_tgts  = get_attention_targets(model)

    lora_cfg = build_lora_config(
        mamba_targets=mamba_tgts,
        attn_targets=attn_tgts,
        rank_m=args.rank,
        rank_a=args.rank,
        alpha_m=alpha,
        alpha_a=alpha,
        dropout=args.dropout,
    )
    model = inject_lora(model, lora_cfg)
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
    elapsed = time.time() - t0
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    train_time = f"{h}:{m:02d}:{s:02d}"
    logger.info("Training complete in %s", train_time)

    # ── Results ───────────────────────────────────────────────────────────────
    evals = [e for e in trainer.state.log_history if "eval_loss" in e]
    best_loss = min(e["eval_loss"] for e in evals) if evals else None
    best_ppl  = compute_perplexity(best_loss) if best_loss else None

    rho_values = [v for _, v in trainer.rho_log]
    rho_avg    = round(float(sum(rho_values) / len(rho_values)), 4) if rho_values else None
    rho_final  = round(float(rho_values[-1]), 4) if rho_values else None

    metrics = {
        "strategy":        "symmetric_lora",
        "model":           args.model,
        "rank":            args.rank,
        "alpha":           alpha,
        **param_counts,
        "best_eval_loss":  round(best_loss, 6) if best_loss else None,
        "best_eval_ppl":   round(best_ppl, 4)  if best_ppl  else None,
        "rho_avg":         rho_avg,
        "rho_final":       rho_final,
        "training_time":   train_time,
    }
    save_metrics(metrics, fig_dir)
    logger.info("Best PPL: %.4f | Params: %.3fM", best_ppl or 0, param_counts["total_M"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_loss_curves(trainer.train_loss_log, trainer.eval_loss_log,
                     f"Symmetric LoRA r={args.rank} — WikiText-2", fig_dir)
    plot_ppl_curve(trainer.eval_loss_log,
                   f"Symmetric LoRA r={args.rank} — WikiText-2 PPL", fig_dir)
    plot_rho(trainer.rho_log,
             f"Gradient norm ratio ρ — Symmetric LoRA r={args.rank}", fig_dir)
    plot_lr(trainer.lr_log, fig_dir)

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Adapter saved → %s", out_dir)


if __name__ == "__main__":
    main()
