"""Phase 2: fine-tune VietAI/vit5-base bằng LoRA với Seq2SeqTrainer.

Ví dụ:
    # Smoke test CPU
    python -m vnsum.models.train --config configs/train.yaml --max_steps 5 --limit 32
    # Colab (checkpoint trên Drive, tự resume khi chạy lại)
    python -m vnsum.models.train --config configs/train.yaml \\
        --output_dir /content/drive/MyDrive/vn-longsum/runs/vit5-lora
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from vnsum.models.train_config import apply_cli_overrides, load_train_config

logger = logging.getLogger("vnsum.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ViT5 + LoRA (phase 2)")
    p.add_argument("--config", required=True)
    p.add_argument("--max_steps", type=int, default=None, help="dừng sau N optimizer step (smoke test)")
    p.add_argument("--limit", type=int, default=None, help="tổng số mẫu train; bật profile `smoke` trong config")
    p.add_argument("--output_dir", default=None, help="override paths.output_dir (vd. thư mục trên Google Drive)")
    p.add_argument("--processed_dir", default=None, help="override paths.processed_dir")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None, help="override precision")
    p.add_argument("--resume_from_checkpoint", default=None, help="auto | none | đường dẫn checkpoint")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def resolve_resume(resume: str | None, output_dir: str) -> str | None:
    """``auto`` -> checkpoint mới nhất trong output_dir (nếu có); ``none`` -> None; còn lại là đường dẫn."""
    if resume in (None, "none", "None", False):
        return None
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        if not os.path.isdir(output_dir):
            return None
        last = get_last_checkpoint(output_dir)
        if last:
            logger.info("Resume từ checkpoint mới nhất: %s", last)
        return last
    if not os.path.isdir(resume):
        raise FileNotFoundError(f"Không thấy checkpoint {resume}")
    return resume


def build_trainer_class():
    from transformers import Seq2SeqTrainer

    import torch

    from vnsum.models.callbacks import NonFiniteLossError, nonfinite_message

    class GuardedSeq2SeqTrainer(Seq2SeqTrainer):
        """Kiểm tra loss từng micro-step: NaN/inf -> dừng ngay, không đợi tới lần log."""

        precision = "fp32"

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not torch.isfinite(loss.detach()).all():
                raise NonFiniteLossError(nonfinite_message(float(loss.detach().float().mean()), self.state.global_step, self.precision))
            return loss

    return GuardedSeq2SeqTrainer


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub.utils._http"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = apply_cli_overrides(
        load_train_config(args.config),
        limit=args.limit,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        precision=args.precision,
        resume=args.resume_from_checkpoint,
    )
    if cfg["limit"]:
        logger.info("Smoke mode: --limit %d, đã áp dụng profile `smoke` từ config", cfg["limit"])

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, set_seed

    from vnsum.data.tokenization import load_tokenizer
    from vnsum.models.callbacks import NonFiniteLossCallback, RougeEvalCallback
    from vnsum.models.data import load_eval_frame, load_train_frame, tokenize_frame
    from vnsum.models.lora import count_parameters, linear_module_names, report_modules, resolve_target_modules

    seed = int(cfg["seed"])
    set_seed(seed)
    precision = cfg["precision"]
    use_cuda = torch.cuda.is_available()
    if precision in ("fp16", "bf16") and not use_cuda:
        raise SystemExit(f"precision={precision} cần GPU CUDA. Trên CPU hãy dùng fp32.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            f"GPU {torch.cuda.get_device_name(0)} không hỗ trợ bf16 (vd. T4). Dùng precision fp32 (khuyến nghị) hoặc fp16."
        )
    if use_cuda:
        name = torch.cuda.get_device_name(0)
        logger.info("GPU: %s | bf16 supported: %s | precision: %s", name, torch.cuda.is_bf16_supported(), precision)
    else:
        logger.info("Không có GPU -> chạy CPU, precision fp32")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(cfg["hf"]["token_env"]) or None
    model_name = cfg["model"]["name_or_path"]

    # ---------------- dữ liệu ----------------
    tokenizer = load_tokenizer(model_name, cfg["hf"]["token_env"])
    train_df = load_train_frame(cfg)
    eval_df = load_eval_frame(cfg)
    if cfg["limit"]:
        eval_df = eval_df.head(max(1, min(len(eval_df), cfg["limit"])))
    train_ds = tokenize_frame(train_df, tokenizer, cfg)
    eval_ds = tokenize_frame(eval_df, tokenizer, cfg)

    # ---------------- model + LoRA ----------------
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token, dtype=torch.float32)
    model.config.use_cache = False  # không tương thích gradient checkpointing khi train
    names = linear_module_names(model)
    report_modules(names, out_dir)
    lora_cfg = cfg["lora"]
    targets = resolve_target_modules(names, list(lora_cfg["target_modules"]), bool(lora_cfg.get("include_ffn")))
    print(f"LoRA target_modules: {targets}")
    if cfg["training"].get("gradient_checkpointing"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=lora_cfg.get("bias", "none"),
            target_modules=targets,
        ),
    )
    trainable, total = count_parameters(model)
    print(f"Tham số trainable: {trainable:,} / tổng {total:,} ({100 * trainable / total:.3f}%)")

    # ---------------- trainer ----------------
    training_kwargs = dict(cfg["training"])
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        data_seed=seed,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
        **training_kwargs,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=None, label_pad_token_id=-100, pad_to_multiple_of=8 if precision == "fp16" else None
    )
    ev = cfg["eval"]
    rouge_cb = RougeEvalCallback(
        eval_df[["source", "summary"]].to_dict("records"),
        tokenizer,
        source_prefix=cfg["data"].get("source_prefix") or "",
        max_source_length=int(cfg["data"]["max_source_length"]),
        batch_size=int(ev["batch_size"]),
        generation=ev["generation"],
        output_file=out_dir / "rouge_subset_history.jsonl",
        autocast_dtype=precision if precision in ("fp16", "bf16") else None,
    )
    trainer_cls = build_trainer_class()
    trainer_cls.precision = precision
    trainer = trainer_cls(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[NonFiniteLossCallback(precision), rouge_cb],
    )
    rouge_cb.trainer = trainer
    # T5.forward có **kwargs nên Trainer tưởng model tự chuẩn hóa loss theo
    # num_items_in_batch và BỎ bước chia cho gradient_accumulation_steps. Nhưng T5
    # vẫn dùng CrossEntropyLoss(mean) -> loss và gradient bị nhân với số bước tích
    # lũy (đã thấy thực tế: loss log ~70 với accumulation 16). Ép Trainer tự chia.
    trainer.model_accepts_loss_kwargs = False

    resume = resolve_resume(cfg["resume"], str(out_dir))
    result = trainer.train(resume_from_checkpoint=resume)

    final_dir = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    info = {
        "model": model_name,
        "precision": precision,
        "input_strategy": cfg["data"]["input_strategy"],
        "target_modules": targets,
        "trainable_params": trainable,
        "total_params": total,
        "train_samples": len(train_ds),
        "global_step": trainer.state.global_step,
        "train_loss": result.training_loss,
        "resumed_from": resume,
        "config": cfg,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nXong. global_step={trainer.state.global_step}, train_loss={result.training_loss:.4f}")
    print(f"Adapter + tokenizer: {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
