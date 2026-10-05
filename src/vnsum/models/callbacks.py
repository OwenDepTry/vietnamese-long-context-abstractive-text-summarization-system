"""Callback cho Seq2SeqTrainer: chặn loss NaN/inf và ROUGE trên tập con val mỗi epoch."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from vnsum.models.rouge import compute_rouge

logger = logging.getLogger(__name__)


class NonFiniteLossError(RuntimeError):
    """Loss = NaN/inf trong lúc train."""


def nonfinite_message(value: float, step: int, precision: str) -> str:
    hint = (
        "Họ T5 rất dễ overflow ở fp16 (giá trị activation vượt 65504). Chạy lại với `precision: fp32` "
        "(mặc định) hoặc `--precision fp32`; resume từ checkpoint gần nhất TRƯỚC khi NaN vẫn dùng được."
        if precision == "fp16"
        else "bf16 có cùng dải giá trị với fp32 nên hiếm khi overflow: thử giảm learning_rate, kiểm tra dữ liệu, "
        "hoặc chạy lại fp32 để đối chiếu."
        if precision == "bf16"
        else "Đang chạy fp32 nên ít khả năng do overflow: thử giảm learning_rate (vd. 3e-4 -> 1e-4), "
        "kiểm tra dữ liệu (summary rỗng / quá dài) và max_grad_norm."
    )
    return f"Loss không hữu hạn ({value}) tại global step {step}, precision={precision}. Đã dừng train. {hint}"


class NonFiniteLossCallback(TrainerCallback):
    """Dừng sớm và báo lỗi rõ ràng khi loss được log là NaN/inf."""

    def __init__(self, precision: str) -> None:
        self.precision = precision

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        for key in ("loss", "eval_loss"):
            value = logs.get(key)
            if value is not None and not math.isfinite(float(value)):
                control.should_training_stop = True
                raise NonFiniteLossError(nonfinite_message(value, state.global_step, self.precision))
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            logger.warning(
                "grad_norm=%s tại step %d (fp16 GradScaler sẽ bỏ qua step này; lặp lại nhiều lần là dấu hiệu "
                "overflow).", grad_norm, state.global_step,
            )
        return control


class RougeEvalCallback(TrainerCallback):
    """Cuối mỗi epoch: generate trên tập con val và log ROUGE (predict_with_generate vẫn tắt).

    Gán ``callback.trainer = trainer`` sau khi tạo Trainer.
    """

    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: Any,
        *,
        source_prefix: str,
        max_source_length: int,
        batch_size: int,
        generation: dict[str, Any],
        output_file: str | Path,
        autocast_dtype: str | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.prefix = source_prefix or ""
        self.max_source_length = int(max_source_length)
        self.batch_size = int(batch_size)
        self.generation = dict(generation)
        self.output_file = Path(output_file)
        # Generate dưới autocast cùng precision với lúc train (fp32 thuần trên GPU laptop rất chậm).
        self.autocast_dtype = autocast_dtype
        self.trainer = None

    def _generate(self, model: Any, device: Any) -> list[str]:
        import torch

        preds: list[str] = []
        for start in range(0, len(self.records), self.batch_size):
            batch = self.records[start : start + self.batch_size]
            enc = self.tokenizer(
                [self.prefix + r["source"] for r in batch],
                max_length=self.max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            use_amp = self.autocast_dtype is not None and device.type == "cuda"
            amp_dtype = torch.bfloat16 if self.autocast_dtype == "bf16" else torch.float16
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    use_cache=True,
                    **self.generation,
                )
            preds.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return preds

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or not self.records:
            return control
        model = self.trainer.model
        was_training = model.training
        model.eval()
        t0 = time.time()
        try:
            preds = self._generate(model, args.device)
        finally:
            if was_training:
                model.train()
        refs = [r["summary"] for r in self.records]
        scores = compute_rouge(preds, refs)
        metrics = {f"eval_subset_{k}": v for k, v in scores.items()}
        metrics["eval_subset_n"] = len(preds)
        metrics["eval_subset_gen_seconds"] = round(time.time() - t0, 1)
        self.trainer.log(metrics)
        print(f"\n[ROUGE val subset, epoch {state.epoch:.2f}, step {state.global_step}] {scores}")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("a", encoding="utf-8") as f:
            sample = {"source": self.records[0]["source"][:300], "reference": refs[0], "prediction": preds[0]}
            f.write(
                json.dumps(
                    {"epoch": state.epoch, "step": state.global_step, **metrics, "sample": sample},
                    ensure_ascii=False,
                )
                + "\n"
            )
        return control
