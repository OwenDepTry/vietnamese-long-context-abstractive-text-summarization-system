"""Tập đánh giá: lấy mẫu cố định (seed) từ split test của phase 1."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ["id", "document", "summary", "input_text", "document_tokens"]


def load_eval_set(cfg: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """DataFrame: id, dataset, document, summary, input_text, document_tokens, is_long.

    ``limit`` (smoke test) là tổng số mẫu, chia đều cho các dataset.
    """
    es = cfg["eval_set"]
    seed = int(cfg["seed"])
    per = es.get("samples_per_dataset")
    if limit:
        per = max(1, int(limit) // len(es["datasets"]))
    frames = []
    for name in es["datasets"]:
        path = Path(cfg["paths"]["processed_dir"]) / name / f"{es['split']}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Không thấy {path}. Chạy phase 1 trước (scripts/prepare_data.py).")
        df = pd.read_parquet(path)
        missing = [c for c in _COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} thiếu cột {missing}; chạy lại phase 1 với strategy extractive_filter.")
        df = df[_COLUMNS].copy()
        if per is not None and per < len(df):
            df = df.sample(n=int(per), random_state=seed)
        df.insert(1, "dataset", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["is_long"] = out["document_tokens"] > int(es["long_doc_threshold_tokens"])
    logger.info(
        "Tập đánh giá: %d mẫu %s, %d mẫu dài (> %d token)",
        len(out), out["dataset"].value_counts().to_dict(), int(out["is_long"].sum()), es["long_doc_threshold_tokens"],
    )
    return out
