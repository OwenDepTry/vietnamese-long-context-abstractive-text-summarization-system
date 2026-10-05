"""Đọc Parquet của phase 1 và tạo dataset đã tokenize cho Seq2SeqTrainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cột nguồn theo chiến lược long-context (phase 1).
_SOURCE_COLUMN = {"extractive_filter": "input_text", "truncate": "document"}


def read_split(processed_dir: str | Path, dataset: str, split: str, input_strategy: str) -> pd.DataFrame:
    """Đọc một split, trả DataFrame ``[id, dataset, source, summary]``."""
    path = Path(processed_dir) / dataset / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Chạy phase 1 trước: python scripts/prepare_data.py --config configs/data.yaml"
        )
    df = pd.read_parquet(path)
    col = _SOURCE_COLUMN[input_strategy]
    if col not in df.columns:
        raise ValueError(
            f"{path} không có cột {col!r} (cần cho input_strategy={input_strategy}). "
            "Chạy lại phase 1 với long_context.strategy=extractive_filter."
        )
    out = pd.DataFrame(
        {"id": df["id"], "dataset": dataset, "source": df[col].fillna(""), "summary": df["summary"].fillna("")}
    )
    return out[(out["source"].str.len() > 0) & (out["summary"].str.len() > 0)].reset_index(drop=True)


def load_train_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Gộp split train của mọi dataset, xáo trộn theo seed, áp dụng giới hạn mẫu."""
    data = cfg["data"]
    seed = int(cfg["seed"])
    limit = cfg.get("limit")
    frames = [
        read_split(cfg["paths"]["processed_dir"], name, "train", data["input_strategy"]) for name in data["datasets"]
    ]
    if limit:
        per = max(1, limit // len(frames))
        frames = [f.sample(n=min(per, len(f)), random_state=seed) for f in frames]
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cap = data.get("max_train_samples")
    if cap and not limit:
        df = df.iloc[: int(cap)].reset_index(drop=True)
    logger.info("Train: %d mẫu (%s)", len(df), df["dataset"].value_counts().to_dict())
    return df


def load_eval_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Tập con validation cố định (theo seed) để tính eval loss + ROUGE mỗi epoch."""
    data = cfg["data"]
    n = int(cfg["eval"]["rouge_samples_per_dataset"])
    frames = []
    for name in data["datasets"]:
        f = read_split(cfg["paths"]["processed_dir"], name, "validation", data["input_strategy"])
        frames.append(f.sample(n=min(n, len(f)), random_state=int(cfg["seed"])))
    df = pd.concat(frames, ignore_index=True)
    logger.info("Eval subset: %d mẫu (%s)", len(df), df["dataset"].value_counts().to_dict())
    return df


def tokenize_frame(df: pd.DataFrame, tokenizer: Any, cfg: dict[str, Any]):
    """Tokenize thành ``datasets.Dataset`` chỉ gồm input_ids, attention_mask, labels."""
    from datasets import Dataset

    data = cfg["data"]
    prefix = data.get("source_prefix") or ""
    max_src, max_tgt = int(data["max_source_length"]), int(data["max_target_length"])

    def _tok(batch: dict[str, list]) -> dict[str, list]:
        enc = tokenizer([prefix + s for s in batch["source"]], max_length=max_src, truncation=True)
        labels = tokenizer(text_target=batch["summary"], max_length=max_tgt, truncation=True)
        enc["labels"] = labels["input_ids"]
        # Độ dài tính luôn ở đây (cột int của Arrow) thay vì đọc lại cả cột input_ids sau đó.
        enc["_src_len"] = [len(x) for x in enc["input_ids"]]
        enc["_tgt_len"] = [len(x) for x in enc["labels"]]
        return enc

    ds = Dataset.from_pandas(df[["source", "summary"]], preserve_index=False)
    ds = ds.map(_tok, batched=True, remove_columns=["source", "summary"], num_proc=int(data.get("num_proc", 1)) or None)
    src_len = np.asarray(ds["_src_len"])
    tgt_len = np.asarray(ds["_tgt_len"])
    src_cap = int((src_len >= max_src).sum())
    tgt_cap = int((tgt_len >= max_tgt).sum())
    ds = ds.remove_columns(["_src_len", "_tgt_len"])
    logger.info(
        "Tokenize %d mẫu: %.2f%% source chạm max_source_length=%d, %.2f%% summary chạm max_target_length=%d",
        len(ds), 100 * src_cap / max(1, len(ds)), max_src, 100 * tgt_cap / max(1, len(ds)), max_tgt,
    )
    return ds
