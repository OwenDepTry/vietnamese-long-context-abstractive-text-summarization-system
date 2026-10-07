"""Phần logic của benchmark không phụ thuộc torch / onnxruntime: các lượt chạy, thống kê latency,
so sánh chất lượng và sinh ``benchmark.md``. Mọi con số trong report lấy từ kết quả đo truyền vào."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# "before" của mỗi thiết bị là PyTorch; các bản ONNX là "after".
RUNS: dict[str, dict[str, Any]] = {
    "pytorch_cpu": {"backend": "pytorch", "variant": None, "device": "cpu", "label": "PyTorch (trước)"},
    "onnx_fp32_cpu": {"backend": "onnx", "variant": "fp32", "device": "cpu", "label": "ONNX FP32"},
    "onnx_int8_cpu": {"backend": "onnx", "variant": "int8", "device": "cpu", "label": "ONNX INT8 dynamic"},
    "pytorch_cuda": {"backend": "pytorch", "variant": None, "device": "cuda", "label": "PyTorch (trước)"},
    "onnx_fp32_cuda": {"backend": "onnx", "variant": "fp32", "device": "cuda", "label": "ONNX FP32"},
    "onnx_fp16_cuda": {"backend": "onnx", "variant": "fp16", "device": "cuda", "label": "ONNX FP16"},
}
BASELINE = {"cpu": "pytorch_cpu", "cuda": "pytorch_cuda"}


def runs_for(device: str) -> list[str]:
    if device == "all":
        return list(RUNS)
    return [k for k, v in RUNS.items() if v["device"] == device]


def select_documents(eval_df, n: int, seed: int):
    """n tài liệu cố định theo seed từ tập đánh giá phase 3 (trộn cả hai dataset)."""
    n = min(int(n), len(eval_df))
    return eval_df.sample(n=n, random_state=int(seed)).reset_index(drop=True)


# --------------------------------------------------------------------------- thống kê
def latency_stats(latency_ms: Sequence[float], new_tokens: Sequence[int], generate_ms: Sequence[float]) -> dict[str, float]:
    lat = np.asarray(latency_ms, dtype=float)
    if lat.size == 0:
        return {}
    gen_s = float(np.sum(generate_ms)) / 1000
    return {
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
        "max_ms": float(lat.max()),
        "docs_per_s": float(lat.size / (lat.sum() / 1000)),
        "tokens_per_s": float(np.sum(new_tokens) / gen_s) if gen_s > 0 else float("nan"),
        "mean_new_tokens": float(np.mean(new_tokens)),
    }


def bucket_edges_labels(edges: Sequence[int]) -> list[tuple[int, int, str]]:
    out, lo = [], 0
    for e in edges:
        out.append((lo, int(e), f"≤{int(e)}" if lo == 0 else f"{lo + 1}–{int(e)}"))
        lo = int(e)
    out.append((lo, math.inf, f">{lo}"))
    return out


def bucket_latency(latency_ms: Sequence[float], input_tokens: Sequence[int], edges: Sequence[int]) -> list[dict[str, Any]]:
    lat, tok = np.asarray(latency_ms, float), np.asarray(input_tokens, float)
    rows = []
    for lo, hi, label in bucket_edges_labels(edges):
        m = (tok > lo) & (tok <= hi)
        rows.append({"bucket": label, "n": int(m.sum()),
                     "p50_ms": float(np.percentile(lat[m], 50)) if m.any() else None,
                     "p95_ms": float(np.percentile(lat[m], 95)) if m.any() else None})
    return rows


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    from vnsum.eval.metrics import rouge_per_sample

    scores = rouge_per_sample(list(predictions), list(references))
    return float(np.mean([s["rougeL"] for s in scores])) if scores else float("nan")


def quality_table(results: dict[str, dict[str, Any]], references: Sequence[str], drop_warning: float) -> list[dict[str, Any]]:
    """ROUGE-L từng lượt chạy và hiệu so với PyTorch cùng thiết bị, trên cùng tài liệu."""
    rl = {k: rouge_l(r["predictions"], references) for k, r in results.items() if r.get("status") == "ok"}
    rows = []
    for k, r in results.items():
        if k not in rl:
            continue
        base = BASELINE[RUNS[k]["device"]]
        row = {"run": k, "rougeL": rl[k], "baseline": base if base != k else None,
               "delta": None, "identical": None, "warning": False}
        if base != k and base in rl:
            row["delta"] = rl[k] - rl[base]
            same = sum(a == b for a, b in zip(r["predictions"], results[base]["predictions"]))
            row["identical"] = same
            row["warning"] = row["delta"] < -float(drop_warning)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- report
def _f(v: Any, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _hardware_lines(info: dict[str, Any]) -> list[str]:
    gpus = info.get("gpus") or []
    gpu = "; ".join(f"{g['name']} ({g['memory_total_mb']} MiB, driver {g['driver']})" for g in gpus) or "không có / không đọc được"
    libs = ", ".join(f"{k} {v}" for k, v in (info.get("libraries") or {}).items())
    return [
        f"- **CPU:** {info.get('cpu')} — {info.get('logical_cores')} luồng logic; PyTorch dùng {info.get('torch_num_threads', '—')} luồng",
        f"- **RAM:** {_f(info.get('ram_gb'), 1)} GB",
        f"- **GPU:** {gpu}; CUDA của torch: {info.get('torch_cuda') or '—'}",
        f"- **Hệ điều hành:** {info.get('os')} ({info.get('machine')})",
        f"- **ONNX Runtime providers:** {', '.join(info.get('onnxruntime_providers') or []) or '—'}",
        f"- **Thư viện:** {libs}",
    ]


def write_report(results: dict[str, dict[str, Any]], references: Sequence[str], cfg: dict[str, Any],
                 info: dict[str, Any], out_path: Path, *, command: str, gen_kwargs: dict[str, Any]) -> list[str]:
    """Ghi benchmark.md; trả về danh sách cảnh báo (để in ra console)."""
    b = cfg["benchmark"]
    target = float(b["latency_target_ms"])
    ok = {k: r for k, r in results.items() if r.get("status") == "ok"}
    stats = {k: latency_stats(r["latency_ms"], r["new_tokens"], r["generate_ms"]) for k, r in ok.items()}
    quality = {q["run"]: q for q in quality_table(results, references, float(b["rouge_drop_warning"]))} if ok else {}
    any_r = next(iter(ok.values()), None) or next(iter(results.values()), {})
    warnings: list[str] = []

    lines = [
        "# Phase 4 — Benchmark inference",
        "",
        f"Sinh tự động lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `scripts/benchmark.py`. "
        "Mọi con số dưới đây là số đo thật của lần chạy này (file JSON từng lượt chạy nằm cạnh file này).",
        "",
        f"Lệnh: `{command}`",
        "",
        "## Điều kiện đo",
        "",
        f"- **Tài liệu:** {any_r.get('n', '—')} tài liệu test cố định (seed {cfg.get('seed', 42)}) lấy từ tập đánh giá phase 3, "
        f"giống nhau cho mọi lượt chạy; {b['warmup']} lần warmup (không tính) trước khi đo.",
        f"- **Input:** chiến lược `{b['strategy']}` (BM25 extractive filter ≤ {cfg['max_input_tokens']} token như lúc train / phase 3).",
        f"- **Generation:** batch 1, num_beams={gen_kwargs['num_beams']}, max_new_tokens={gen_kwargs['max_new_tokens']} "
        f"(mức `{b['length']}`), no_repeat_ngram_size={gen_kwargs['no_repeat_ngram_size']}, early_stopping={gen_kwargs['early_stopping']}.",
        "- **Latency:** thời gian end-to-end một tài liệu (chuẩn hóa + chọn câu + tokenize + generate + decode), đo bằng "
        "`time.perf_counter`; throughput = số tài liệu / tổng thời gian khi chạy tuần tự.",
        "- **Peak RAM:** RAM đỉnh của process chạy lượt đó (mỗi lượt chạy trong một process riêng). "
        "**Peak VRAM:** mức tăng `memory.used` của GPU (nvidia-smi) so với trước khi nạp model.",
        "",
        "## Phần cứng",
        "",
        *_hardware_lines(info),
    ]

    for device, title in (("cpu", "CPU"), ("cuda", "GPU")):
        keys = [k for k in RUNS if RUNS[k]["device"] == device and k in results]
        if not keys:
            continue
        base = BASELINE[device]
        rows = []
        for k in keys:
            r, s, q = results[k], stats.get(k, {}), quality.get(k, {})
            if r.get("status") != "ok":
                rows.append([f"`{k}`", RUNS[k]["label"], "—", "—", "—", "—", "—", "—", "—", "—", f"bỏ qua: {r.get('reason', '')}"])
                continue
            speed = (stats[base]["p50_ms"] / s["p50_ms"]) if base in stats and k != base else None
            mem = _f(r.get("peak_vram_mb")) if device == "cuda" else _f(r.get("peak_rss_mb"))
            delta = q.get("delta")
            rows.append([f"`{k}`", RUNS[k]["label"], r.get("dtype", "—"), _f(s["p50_ms"]), _f(s["p95_ms"]),
                         _f(s["docs_per_s"], 2), _f(s["tokens_per_s"], 1), mem, _f(r.get("disk_mb")),
                         _f(q.get("rougeL"), 2) + ("" if delta is None else f" ({delta:+.2f})"),
                         "1.00×" if k == base else (f"{speed:.2f}×" if speed else "—")])
        mem_label = "Peak VRAM (MB)" if device == "cuda" else "Peak RAM (MB)"
        lines += ["", f"## Kết quả {title}", "",
                  _table(["Lượt chạy", "Bản", "dtype", "p50 (ms)", "p95 (ms)", "Tài liệu/s", "Token sinh/s", mem_label,
                          "Trên đĩa (MB)", "ROUGE-L (Δ so với PyTorch)", "Tăng tốc p50"], rows)]
        if device == "cuda":
            extra = [f"`{k}`: torch.cuda.max_memory_allocated = {_f(results[k].get('torch_max_allocated_mb'))} MB"
                     for k in keys if results[k].get("torch_max_allocated_mb") is not None]
            if extra:
                lines += ["", "PyTorch allocator: " + "; ".join(extra) + "."]
        if device == "cpu":
            threads = [f"`{k}` {results[k].get('threads')}" for k in keys if results[k].get("threads")]
            if threads:
                lines += ["", "Số luồng CPU: " + ", ".join(threads) + "."]

    # ----- latency theo độ dài input
    edges = b.get("length_buckets", [256, 512])
    if ok:
        labels = [lab for _, _, lab in bucket_edges_labels(edges)]
        rows = []
        for k in ok:
            bl = bucket_latency(ok[k]["latency_ms"], ok[k]["input_tokens"], edges)
            rows.append([f"`{k}`"] + [f"{_f(x['p50_ms'])} / {_f(x['p95_ms'])} (n={x['n']})" for x in bl])
        lines += ["", "## Latency theo độ dài input (p50 / p95, ms)", "",
                  "Độ dài = số token ViT5 thực sự đưa vào model (sau extractive filter, gồm `</s>`).", "",
                  _table(["Lượt chạy"] + [f"input {lab} token" for lab in labels], rows)]

    # ----- mục tiêu
    lines += ["", f"## Mục tiêu latency < {target:.0f} ms", ""]
    if not ok:
        lines.append("Không có lượt chạy nào thành công.")
    else:
        rows, met_any = [], []
        for k in ok:
            s = stats[k]
            rows.append([f"`{k}`", _f(s["p50_ms"]), "đạt" if s["p50_ms"] < target else "không đạt",
                         _f(s["p95_ms"]), "đạt" if s["p95_ms"] < target else "không đạt"])
            for x in bucket_latency(ok[k]["latency_ms"], ok[k]["input_tokens"], edges):
                if x["n"] and x["p95_ms"] is not None and x["p95_ms"] < target:
                    met_any.append(f"`{k}` với input {x['bucket']} token (p95 {_f(x['p95_ms'])} ms, n={x['n']})")
        lines += [_table(["Lượt chạy", "p50 (ms)", "p50 < mục tiêu", "p95 (ms)", "p95 < mục tiêu"], rows), ""]
        full = [k for k in ok if stats[k]["p95_ms"] < target]
        if full:
            lines.append(f"**Đạt mục tiêu ở p95 trên toàn bộ tài liệu đo:** {', '.join(f'`{k}`' for k in full)} "
                         f"(beam {gen_kwargs['num_beams']}, max_new_tokens {gen_kwargs['max_new_tokens']}, batch 1).")
        else:
            lines.append(f"**Không lượt chạy nào đạt p95 < {target:.0f} ms trên toàn bộ tài liệu đo** "
                         f"(beam {gen_kwargs['num_beams']}, max_new_tokens {gen_kwargs['max_new_tokens']}, batch 1).")
        lines.append("")
        lines.append("Theo nhóm độ dài input, p95 < mục tiêu ở: " + ("; ".join(met_any) if met_any else "không nhóm nào") + ".")

    # ----- chất lượng
    lines += ["", "## Suy giảm chất lượng (ROUGE-L, cùng tài liệu)", ""]
    qrows = []
    for k, q in quality.items():
        if q["baseline"] is None:
            continue
        flag = "⚠ giảm quá ngưỡng" if q["warning"] else "trong ngưỡng"
        qrows.append([f"`{k}`", f"`{q['baseline']}`", _f(q["rougeL"], 2), _f(quality[q["baseline"]]["rougeL"], 2),
                      f"{q['delta']:+.2f}", f"{q['identical']}/{len(ok[k]['predictions'])}", flag])
        if q["warning"]:
            warnings.append(f"{k}: ROUGE-L thấp hơn {q['baseline']} {abs(q['delta']):.2f} điểm (ngưỡng {b['rouge_drop_warning']}).")
    if qrows:
        lines += [f"Ngưỡng cảnh báo: giảm quá {b['rouge_drop_warning']} điểm ROUGE-L so với PyTorch cùng thiết bị.", "",
                  _table(["Bản", "So với", "ROUGE-L", "ROUGE-L PyTorch", "Hiệu", "Output giống hệt", "Kết luận"], qrows)]
        n_docs = len(references)
        if n_docs < 30:
            lines += ["", f"Chỉ có {n_docs} tài liệu: hiệu ROUGE-L ở cỡ mẫu này dao động mạnh, chưa đủ để kết luận về chất lượng."]
    else:
        lines.append("Chưa đủ lượt chạy (cần PyTorch và ONNX trên cùng thiết bị) để so sánh.")

    skipped = {k: r for k, r in results.items() if r.get("status") != "ok"}
    if skipped:
        lines += ["", "## Lượt chạy bị bỏ qua", ""] + [f"- `{k}`: {r.get('reason', '')}" for k, r in skipped.items()]

    lines += ["", "## Giới hạn", "",
              "- Đo trên một máy, batch 1, chạy tuần tự; laptop có thể giảm xung khi nóng hoặc chạy pin, nên số đo dao động giữa các lần.",
              "- Peak VRAM đọc từ nvidia-smi là bộ nhớ của cả GPU (trừ mức nền); chương trình khác dùng GPU trong lúc đo sẽ làm sai số này.",
              "- ROUGE-L chỉ đo trùng lặp bề mặt với reference; output giống hệt PyTorch là chỉ báo trực tiếp hơn cho việc chuyển đổi có giữ nguyên hành vi hay không.",
              ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return warnings
