#!/usr/bin/env python
"""Phase 1 pipeline: tải -> chuẩn hóa -> lọc -> khử trùng -> xử lý văn bản dài -> Parquet + thống kê.

Ví dụ:
    python scripts/prepare_data.py --config configs/data.yaml --limit 200

``--limit N`` là TỔNG số mẫu thô tối đa cho cả lần chạy, chia đều cho mọi cặp
(dataset, split) đang bật (vd. 200 mẫu / 6 cặp = 33 mẫu mỗi cặp). Khi có
``--limit``, dữ liệu được đọc streaming nên không phải tải toàn bộ dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # cho phép chạy script khi chưa `pip install -e .`
    sys.path.insert(0, str(_SRC))

from vnsum.config import load_config  # noqa: E402
from vnsum.data.load import load_splits  # noqa: E402
from vnsum.data.long_context import LongContextProcessor  # noqa: E402
from vnsum.data.preprocess import deduplicate, filter_records, normalize_text, word_segment  # noqa: E402
from vnsum.data.tokenization import TokenCounter, build_token_counter  # noqa: E402

logger = logging.getLogger("prepare_data")


def _progress(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except ImportError:
        return iterable


def _num_splits(ds_cfg: dict[str, Any]) -> int:
    return len(ds_cfg.get("splits") or ds_cfg.get("split_ratios") or {})


def _length_stats(values: list[int], threshold: int, pct: float) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), f"p{int(pct)}": float("nan"), "max": 0, "pct_over_threshold": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 1),
        f"p{int(pct)}": round(float(np.percentile(arr, pct)), 1),
        "max": int(arr.max()),
        "pct_over_threshold": round(100.0 * float((arr > threshold).mean()), 2),
    }


def process_dataset(
    name: str, cfg: dict[str, Any], counter: TokenCounter, limit_per_split: int | None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    ds_cfg = cfg["datasets"][name]
    pre_cfg = cfg["preprocess"]
    raw = load_splits(name, cfg, limit_per_split=limit_per_split)

    drop = Counter()
    raw_counts = {s: len(v) for s, v in raw.items()}
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for split, records in raw.items():
        for rec in records:
            rec["document"] = normalize_text(rec["document"], pre_cfg, desegment_text=ds_cfg.get("desegment", False))
            rec["summary"] = normalize_text(rec["summary"], pre_cfg, desegment_text=ds_cfg.get("desegment", False))
        kept, reasons = filter_records(records, cfg["filter"])
        drop.update(reasons)
        cleaned[split] = kept
    cleaned, dup = deduplicate(cleaned, cfg["filter"]["dedup"])
    drop.update(dup)

    processor = LongContextProcessor(cfg["long_context"], counter, cfg["sentence_split"])
    ws_cfg = pre_cfg["word_segmentation"]
    batch = int(cfg["tokenizer"].get("batch_size", 256))
    frames: dict[str, pd.DataFrame] = {}
    for split, records in cleaned.items():
        df = pd.DataFrame(records, columns=["id", "document", "summary"])
        df["document_tokens"] = counter.count_many(df["document"].tolist(), batch_size=batch)
        df["summary_tokens"] = counter.count_many(df["summary"].tolist(), batch_size=batch)
        lc = [processor(d) for d in _progress(df["document"], len(df), f"{name}/{split} long-context")]
        if lc:
            df = pd.concat([df, pd.DataFrame(lc, index=df.index)], axis=1)
        if ws_cfg.get("enabled", False):
            # Cột riêng cho PhoBERT; document/summary (input ViT5) giữ nguyên.
            for col in ws_cfg.get("columns", []):
                df[f"{col}_ws"] = [word_segment(t, ws_cfg["backend"]) for t in df[col]]
        frames[split] = df

    info = {"raw_counts": raw_counts, "dropped": dict(sorted(drop.items()))}
    return frames, info


def summarize(name: str, frames: dict[str, pd.DataFrame], info: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    st = cfg["stats"]
    threshold, pct = int(st["long_doc_threshold_tokens"]), float(st["percentile"])
    strategy = cfg["long_context"]["strategy"]
    label = strategy
    if strategy == "extractive_filter":
        ex = cfg["long_context"]["extractive_filter"]
        label = f"extractive_filter[{ex['ranker']}, {ex['selection']}]"
    out: dict[str, Any] = {"dataset": name, "strategy": label, **info, "splits": {}}
    for split, df in frames.items():
        s: dict[str, Any] = {
            "num_samples": int(len(df)),
            "document_tokens": _length_stats(df["document_tokens"].tolist(), threshold, pct),
            "summary_tokens": _length_stats(df["summary_tokens"].tolist(), threshold, pct),
        }
        if len(df) and strategy == "extractive_filter":
            s["input_tokens_max"] = int(df["input_tokens"].max())
            s["num_filtered"] = int(df["was_filtered"].sum())
            filtered = df.loc[df["was_filtered"], "input_tokens"]
            if len(filtered):
                s["filtered_input_tokens_mean"] = round(float(filtered.mean()), 1)
            s["num_truncated"] = int(df["was_truncated"].sum())
        if len(df) and strategy == "hierarchical":
            s["chunks_mean"] = round(float(df["num_chunks"].mean()), 2)
            s["chunk_tokens_max"] = int(max((max(c) for c in df["chunk_tokens"] if len(c)), default=0))
        out["splits"][split] = s
    return out


def print_report(all_stats: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    pct = int(cfg["stats"]["percentile"])
    thr = int(cfg["stats"]["long_doc_threshold_tokens"])
    tok = cfg["tokenizer"]["name_or_path"]
    header = (
        f"{'dataset':<14} {'split':<10} {'n':>7} {'doc_mean':>9} {f'doc_p{pct}':>8} "
        f"{'sum_mean':>9} {f'sum_p{pct}':>8} {f'%doc>{thr}':>10}"
    )
    print(f"\n=== Thống kê (đơn vị: token của {tok}) ===")
    print(header)
    print("-" * len(header))
    for ds in all_stats:
        for split, s in ds["splits"].items():
            d, m = s["document_tokens"], s["summary_tokens"]
            print(
                f"{ds['dataset']:<14} {split:<10} {s['num_samples']:>7} {d['mean']:>9} {d[f'p{pct}']:>8} "
                f"{m['mean']:>9} {m[f'p{pct}']:>8} {d['pct_over_threshold']:>10}"
            )
    print()
    for ds in all_stats:
        print(f"[{ds['dataset']}] mẫu thô: {ds['raw_counts']} | bị loại: {ds['dropped'] or '{}'}")
        for split, s in ds["splits"].items():
            extra = {k: v for k, v in s.items() if k not in ("num_samples", "document_tokens", "summary_tokens")}
            if extra:
                print(f"  {split}: {ds['strategy']} -> {extra}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="đường dẫn YAML, vd. configs/data.yaml")
    parser.add_argument("--limit", type=int, default=None, help="tổng số mẫu thô tối đa (smoke test)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    enabled = [n for n, d in cfg["datasets"].items() if d.get("enabled", False)]
    if not enabled:
        logger.error("Không có dataset nào được bật trong config")
        return 1

    limit_per_split = None
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit phải > 0")
        pairs = sum(_num_splits(cfg["datasets"][n]) for n in enabled)
        limit_per_split = max(1, args.limit // pairs)
        logger.info("--limit %d -> tối đa %d mẫu cho mỗi (dataset, split), %d cặp", args.limit, limit_per_split, pairs)

    counter = build_token_counter(cfg)
    out_root = Path(cfg["paths"]["processed_dir"])
    all_stats: list[dict[str, Any]] = []
    for name in enabled:
        frames, info = process_dataset(name, cfg, counter, limit_per_split)
        for split, df in frames.items():
            path = out_root / name / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info("Đã ghi %s (%d dòng)", path, len(df))
        all_stats.append(summarize(name, frames, info, cfg))

    stats_path = Path(cfg["paths"]["stats_file"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"limit": args.limit, "limit_per_split": limit_per_split, "datasets": all_stats}
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(all_stats, cfg)
    print(f"\nĐã lưu thống kê: {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
