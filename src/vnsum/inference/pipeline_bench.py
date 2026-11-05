"""Benchmark đúng đường xử lý đang triển khai (API + UI), không chỉ backend.

``scripts/benchmark.py`` (phase 4) so sánh backend PyTorch / ONNX với input đã lọc extractive cố định. Nó không
đo cái người dùng thật sự chờ: UI gửi 3 request (short, medium, long) cho cùng văn bản, văn bản dài đi qua
hierarchical concat (nhiều lần sinh). Module này mô phỏng đúng luồng đó với config đang dùng, và so sánh các
biến thể (vd. sinh chunk từng cái / theo batch, có / không dùng lại bản tóm tắt chunk) trên cùng tài liệu.

Phần logic không phụ thuộc torch; ``scripts/benchmark_pipeline.py`` lo nạp model, đo bộ nhớ và ghi báo cáo.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

LENGTHS = ("short", "medium", "long")

# Biến thể mặc định: "cũ" = hành vi trước khi có batch + dùng lại chunk; "hiện tại" = config đang dùng.
DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "cũ (chunk từng cái, không dùng lại)": {"chunk_batch_size": 1, "chunk_cache_size": 0},
    "hiện tại (config)": {},
}


def select_mixed(eval_df: pd.DataFrame, n_short: int, n_long: int, seed: int) -> pd.DataFrame:
    """Lấy cố định ``n_short`` văn bản ngắn + ``n_long`` văn bản dài (cột ``is_long``) theo seed.

    Chọn ngẫu nhiên đều thì văn bản dài chỉ ~12%, quá ít để đo đường hierarchical.
    """
    parts = []
    for flag, n in ((False, n_short), (True, n_long)):
        pool = eval_df[eval_df["is_long"].astype(bool) == flag]
        if n > 0 and len(pool):
            parts.append(pool.sample(n=min(int(n), len(pool)), random_state=int(seed)))
    return pd.concat(parts, ignore_index=True) if parts else eval_df.head(0)


def variant_config(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["long_document"].update(overrides)
    return out


def run_ui_flow(summarizer, documents: Sequence[str], lengths: Sequence[str] = LENGTHS,
                ids: Sequence[str] | None = None, clock=time.perf_counter) -> pd.DataFrame:
    """Với mỗi văn bản, gọi lần lượt các mức độ dài như UI. Một dòng mỗi (văn bản, mức độ dài)."""
    rows = []
    ids = list(ids) if ids is not None else [str(i) for i in range(len(documents))]
    for doc_id, doc in zip(ids, documents):
        for order, length in enumerate(lengths):
            t = clock()
            r = summarizer.summarize(doc, length=length)
            rows.append({
                "id": doc_id, "length": length, "order": order, "latency_ms": (clock() - t) * 1000,
                "strategy": r.strategy, "n_chunks": len(r.chunk_summaries) or None,
                "generate_calls": r.generate_calls, "chunk_cache_hit": bool(getattr(r, "chunk_cache_hit", False)),
                "input_tokens": r.input_tokens, "new_tokens": r.new_tokens, "summary": r.summary,
            })
    return pd.DataFrame(rows)


def _pct(x: pd.Series, q: float) -> float:
    return float(np.percentile(x.to_numpy(dtype=float), q)) if len(x) else float("nan")


def summarize_flow(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bảng tổng hợp cho một biến thể.

    * ``per_call``: latency từng request theo chiến lược (direct / extractive / hierarchical).
    * ``per_document``: tổng thời gian chờ cho cả 3 mức độ dài của một văn bản (cái người dùng UI thấy).
    * ``by_chunks``: văn bản hierarchical, theo số chunk.
    """
    per_call = df.groupby("strategy").agg(
        n=("latency_ms", "size"), p50_ms=("latency_ms", lambda s: _pct(s, 50)),
        p95_ms=("latency_ms", lambda s: _pct(s, 95)), mean_calls=("generate_calls", "mean"),
        cache_hit_pct=("chunk_cache_hit", lambda s: 100 * float(np.mean(s))),
    )
    doc = df.groupby("id").agg(total_ms=("latency_ms", "sum"), strategy=("strategy", "first"),
                               n_chunks=("n_chunks", "max"), calls=("generate_calls", "sum"))
    per_document = doc.groupby("strategy").agg(
        n=("total_ms", "size"), p50_ms=("total_ms", lambda s: _pct(s, 50)), p95_ms=("total_ms", lambda s: _pct(s, 95)),
        mean_calls=("calls", "mean"),
    )
    hier = doc[doc["strategy"] == "hierarchical"]
    by_chunks = hier.groupby("n_chunks").agg(
        n=("total_ms", "size"), p50_ms=("total_ms", lambda s: _pct(s, 50)), mean_calls=("calls", "mean"),
    ) if len(hier) else pd.DataFrame()
    return {"per_call": per_call, "per_document": per_document, "by_chunks": by_chunks}


def outputs_identical(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """% request có bản tóm tắt giống hệt nhau giữa hai biến thể (batch có padding có thể lệch nhẹ)."""
    m = a.merge(b, on=["id", "length"], suffixes=("_a", "_b"))
    return 100 * float((m["summary_a"] == m["summary_b"]).mean()) if len(m) else float("nan")


def render_report(results: dict[str, pd.DataFrame], *, header: Sequence[str], memory: dict[str, Any]) -> str:
    lines = list(header) + [""]
    names = list(results)
    tables = {n: summarize_flow(df) for n, df in results.items()}

    lines += ["## Tổng thời gian chờ cho 3 mức độ dài (short + medium + long, như UI)", "",
              "| Biến thể | Chiến lược | Số văn bản | p50 (ms) | p95 (ms) | Lần gọi model TB |", "|---|---|---|---|---|---|"]
    for n in names:
        for strat, r in tables[n]["per_document"].iterrows():
            lines.append(f"| {n} | {strat} | {int(r.n)} | {r.p50_ms:,.0f} | {r.p95_ms:,.0f} | {r.mean_calls:.1f} |")

    lines += ["", "## Từng request", "",
              "| Biến thể | Chiến lược | Số request | p50 (ms) | p95 (ms) | Lần gọi model TB | Dùng lại chunk (%) |",
              "|---|---|---|---|---|---|---|"]
    for n in names:
        for strat, r in tables[n]["per_call"].iterrows():
            lines.append(f"| {n} | {strat} | {int(r.n)} | {r.p50_ms:,.0f} | {r.p95_ms:,.0f} | {r.mean_calls:.1f} | "
                         f"{r.cache_hit_pct:.0f} |")

    lines += ["", "## Văn bản dài theo số chunk (tổng 3 mức độ dài)", "",
              "| Biến thể | Số chunk | Số văn bản | p50 (ms) | Lần gọi model TB |", "|---|---|---|---|---|"]
    for n in names:
        for k, r in tables[n]["by_chunks"].iterrows():
            lines.append(f"| {n} | {int(k)} | {int(r.n)} | {r.p50_ms:,.0f} | {r.mean_calls:.1f} |")

    if len(names) >= 2:
        same = outputs_identical(results[names[0]], results[names[-1]])
        lines += ["", f"Bản tóm tắt giống hệt nhau giữa `{names[0]}` và `{names[-1]}`: **{same:.1f}%** số request. "
                  "Khác biệt (nếu có) đến từ padding khi sinh chunk theo batch, không phải lỗi."]
    if memory:
        lines += ["", "## Bộ nhớ", ""] + [f"- {k}: {v}" for k, v in memory.items()]
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Đo trong một process, gọi `Summarizer` trực tiếp: không gồm thời gian HTTP và hàng đợi của API.",
        "- Request đồng thời chưa được đo: API mặc định `max_concurrency: 1`, nên request thứ hai chờ request đầu.",
        "- Lần gọi đầu cho một văn bản luôn phải sinh chunk; lợi ích dùng lại chỉ có khi cùng văn bản được gọi nhiều "
        "mức độ dài (như UI).",
        "",
    ]
    return "\n".join(lines)
