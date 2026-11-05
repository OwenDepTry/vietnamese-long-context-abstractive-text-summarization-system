#!/usr/bin/env python
"""Benchmark đường xử lý đang triển khai: 3 request (short, medium, long) cho cùng văn bản, như UI.

Khác ``scripts/benchmark.py`` (so backend, input extractive cố định), script này dùng đúng config của API
(``long_document.strategy: auto`` → văn bản dài đi hierarchical concat) và so các biến thể trên CÙNG tài liệu:

* ``cũ``: tóm tắt chunk từng cái một, không dùng lại giữa các mức độ dài (hành vi trước đây).
* ``hiện tại``: theo config (``chunk_batch_size``, ``chunk_cache_size``).

    python scripts/benchmark_pipeline.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
    # nhanh: --n_short 5 --n_long 5

Model chỉ nạp một lần; các biến thể dùng chung backend. Kết quả: results/pipeline_benchmark.md (+ .csv).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import load_inference_config  # noqa: E402
from vnsum.inference.pipeline_bench import (  # noqa: E402
    ABLATION_VARIANTS, DEFAULT_VARIANTS, LENGTHS, render_report, run_ui_flow, select_mixed, variant_config,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--backend", choices=["pytorch", "onnx"], default=None, help="mặc định: backend trong config")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--n_short", type=int, default=20)
    p.add_argument("--n_long", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2, help="số văn bản chạy trước, không tính")
    p.add_argument("--ablation", action="store_true",
                   help="4 biến thể: batch 1/4 × có/không dùng lại chunk (tách nguồn gốc tốc độ và khác biệt output)")
    p.add_argument("--out", default=None, help="mặc định results/pipeline_benchmark.md (--ablation: _ablation.md)")
    args = p.parse_args(argv)

    import pandas as pd

    from vnsum.config import load_config
    from vnsum.eval.config import load_eval_config
    from vnsum.eval.dataset import load_eval_set
    from vnsum.inference.hardware import GpuMemorySampler, peak_rss_mb
    from vnsum.inference.predictor import Summarizer, build_backend

    cfg = load_inference_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    eval_df = load_eval_set(load_eval_config(cfg["benchmark"]["eval_config"]))
    docs = select_mixed(eval_df, args.n_short, args.n_long, int(cfg.get("seed", 42)))
    texts, ids = docs["document"].fillna("").tolist(), docs["id"].astype(str).tolist()

    kwargs = {"model_path": args.model_path}
    if args.device:
        kwargs["device"] = args.device
    backend = build_backend(cfg, args.backend, **kwargs)
    sampler = GpuMemorySampler() if str(backend.device).startswith("cuda") else None
    vram_base = sampler.read_once() if sampler else None
    if sampler:
        sampler.start()

    results: dict[str, pd.DataFrame] = {}
    variants = ABLATION_VARIANTS if args.ablation else DEFAULT_VARIANTS
    for name, overrides in variants.items():
        summ = Summarizer(backend, variant_config(cfg, overrides), data_cfg)
        for t in texts[: args.warmup]:
            summ.summarize(t, length="medium")
        summ._chunk_cache.clear()                     # warmup không được để lại bản tóm tắt chunk
        print(f"[{name}] {len(texts)} văn bản × {len(LENGTHS)} mức độ dài ...", flush=True)
        results[name] = run_ui_flow(summ, texts, LENGTHS, ids=ids)

    peak = sampler.stop() if sampler else None
    ld = cfg["long_document"]
    memory = {"RAM đỉnh của process (MB)": f"{peak_rss_mb():,.0f}" if peak_rss_mb() else "không đo được"}
    if peak is not None and vram_base is not None:
        memory["VRAM tăng thêm, đỉnh (MB, nvidia-smi)"] = f"{peak - vram_base:,.0f}"
    header = [
        "# Benchmark đường xử lý của API (3 mức độ dài như UI)",
        "",
        f"Sinh tự động lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `scripts/benchmark_pipeline.py`. "
        f"{len(texts)} văn bản test ({int((~docs['is_long']).sum())} ngắn, {int(docs['is_long'].sum())} dài > 1024 token), "
        f"backend {backend.name} {backend.dtype} {backend.device}. Config: strategy `{ld['strategy']}`, "
        f"hierarchical_above_tokens {ld['hierarchical_above_tokens']}, reduce `{ld.get('reduce')}`, "
        f"chunk_length `{ld['chunk_length']}`, chunk_batch_size {ld.get('chunk_batch_size', 1)}, "
        f"chunk_cache_size {ld.get('chunk_cache_size', 0)}; beam search như config (không phải streaming).",
    ]
    report = render_report(results, header=header, memory=memory)
    out = Path(args.out or ("results/pipeline_benchmark_ablation.md" if args.ablation else "results/pipeline_benchmark.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    pd.concat([df.assign(variant=n) for n, df in results.items()]).to_csv(out.with_suffix(".csv"), index=False,
                                                                           encoding="utf-8")
    print(report)
    print(f"Đã ghi: {out}, {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
