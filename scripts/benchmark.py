#!/usr/bin/env python
"""Phase 4: benchmark PyTorch vs ONNX (FP32 / INT8 / FP16) trên CPU và GPU.

Chạy trong .venv-onnx, sau scripts/export_onnx.py:
    # Smoke test (5 tài liệu) -> results/smoke/benchmark.md
    python scripts/benchmark.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged --limit 5 --device cpu
    # Đầy đủ (100 tài liệu, CPU + GPU) -> results/benchmark.md
    python scripts/benchmark.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged

Mỗi lượt chạy (backend × thiết bị) chạy trong một process riêng để đo RAM/VRAM đỉnh không lẫn nhau.
Kết quả thô từng lượt: <results>/benchmark/<lượt>.json; output từng tài liệu: predictions.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.bench import RUNS, runs_for, select_documents, write_report  # noqa: E402
from vnsum.inference.config import generation_kwargs, load_inference_config, onnx_dirs, resolve_model_path  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--limit", type=int, default=None, help="số tài liệu (smoke test); kết quả ghi vào <results>/smoke")
    p.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    p.add_argument("--runs", default=None, help=f"chọn lượt chạy, phân cách dấu phẩy: {','.join(RUNS)}")
    p.add_argument("--length", default=None, help="mức độ dài (mặc định benchmark.length)")
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args(argv)


def load_documents(cfg: dict, limit: int | None):
    from vnsum.eval.config import load_eval_config
    from vnsum.eval.dataset import load_eval_set

    eval_cfg = load_eval_config(cfg["benchmark"]["eval_config"])
    df = load_eval_set(eval_cfg)
    df["id"] = df["id"].astype(str)
    return select_documents(df, limit or int(cfg["benchmark"]["n_documents"]), int(cfg.get("seed", 42)))


# --------------------------------------------------------------------------- worker (1 process / lượt chạy)
def run_worker(args, cfg: dict) -> int:
    from vnsum.inference.hardware import (
        ONNX_WEIGHT_PATTERNS, PYTORCH_WEIGHT_PATTERNS, GpuMemorySampler, dir_size_mb, peak_rss_mb,
    )

    key = args.worker
    spec = RUNS[key]
    b = cfg["benchmark"]
    length = args.length or b["length"]
    warmup = int(b["warmup"] if args.warmup is None else args.warmup)
    model_path = resolve_model_path(cfg, args.model_path)
    out = {"run": key, **{k: v for k, v in spec.items() if k != "label"}, "length": length, "strategy": b["strategy"],
           "warmup": warmup, "status": "error"}

    sampler = GpuMemorySampler() if spec["device"] == "cuda" else None
    vram_base = sampler.read_once() if sampler else None
    if sampler:
        sampler.start()
    try:
        docs = load_documents(cfg, args.limit)
        from vnsum.config import load_config
        from vnsum.inference.predictor import OnnxBackend, PyTorchBackend, Summarizer, onnx_session_kwargs

        t = time.perf_counter()
        if spec["backend"] == "pytorch":
            backend = PyTorchBackend(model_path, spec["device"], cfg["pytorch"].get("dtype", "auto"), int(cfg["max_input_tokens"]))
            if spec["device"] == "cuda":
                import torch

                torch.cuda.reset_peak_memory_stats()
            import torch

            out["threads"] = torch.get_num_threads() if spec["device"] == "cpu" else None
            out["disk_mb"] = dir_size_mb(model_path, PYTORCH_WEIGHT_PATTERNS)
        else:
            d = onnx_dirs(cfg, model_path)[spec["variant"]]
            backend = OnnxBackend(d, spec["variant"], spec["device"], **onnx_session_kwargs(cfg),
                                  max_input_tokens=int(cfg["max_input_tokens"]))
            out["threads"] = backend.session_settings["intra_op_num_threads"]
            out["session_settings"] = backend.session_settings
            out["providers"] = backend.providers
            out["disk_mb"] = dir_size_mb(d, ONNX_WEIGHT_PATTERNS)
        out["load_s"] = time.perf_counter() - t
        out["dtype"] = backend.dtype
        summ = Summarizer(backend, cfg, load_config(cfg["data_config"]))

        texts = docs["document"].tolist()
        for i in range(warmup):
            summ.summarize(texts[i % len(texts)], length=length, strategy=b["strategy"])
        rec = {"ids": [], "latency_ms": [], "generate_ms": [], "new_tokens": [], "input_tokens": [], "predictions": []}
        matches = 0
        for row in docs.itertuples():
            r = summ.summarize(row.document, length=length, strategy=b["strategy"])
            rec["ids"].append(row.id)
            rec["latency_ms"].append(r.timings_ms["total_ms"])
            rec["generate_ms"].append(r.timings_ms["generate_ms"])
            rec["new_tokens"].append(r.new_tokens)
            rec["input_tokens"].append(r.model_input_tokens)
            rec["predictions"].append(r.summary)
            matches += int(r.model_input.strip() == str(row.input_text).strip())
        out.update(rec)
        out["n"] = len(rec["ids"])
        out["input_matches_phase3"] = matches
        if spec["backend"] == "pytorch" and spec["device"] == "cuda":
            import torch

            out["torch_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        out["status"] = "ok"
    except Exception as err:
        import traceback

        out["reason"] = f"{type(err).__name__}: {err}"
        out["traceback"] = traceback.format_exc()
    finally:
        peak = sampler.stop() if sampler else None
        out["peak_vram_mb"] = (peak - vram_base) if (peak is not None and vram_base is not None) else None
        out["peak_rss_mb"] = peak_rss_mb()
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if out["status"] == "ok" else 1


# --------------------------------------------------------------------------- orchestrator
def precheck(key: str, cfg: dict, model_path: str) -> str | None:
    """Lý do bỏ qua lượt chạy (None nếu chạy được)."""
    spec = RUNS[key]
    if spec["device"] == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return "torch không thấy GPU (cần torch bản CUDA)"
        except ImportError:
            return "chưa cài torch"
    if spec["backend"] == "onnx":
        d = onnx_dirs(cfg, model_path)[spec["variant"]]
        if not (d.is_dir() and any(d.glob("*.onnx"))):
            hint = " --fp16" if spec["variant"] == "fp16" else ""
            return f"chưa có bản ONNX {spec['variant']} ở {d} (chạy scripts/export_onnx.py{hint})"
        if spec["device"] == "cuda":
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return "onnxruntime không có CUDAExecutionProvider (cài onnxruntime-gpu thay cho onnxruntime)"
    elif not Path(model_path).is_dir():
        return f"không thấy model PyTorch ở {model_path}"
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_inference_config(args.config)
    if args.worker:
        return run_worker(args, cfg)

    from vnsum.inference.hardware import system_info

    b = cfg["benchmark"]
    length = args.length or b["length"]
    gen_kwargs = generation_kwargs(cfg, length)
    model_path = resolve_model_path(cfg, args.model_path)
    results_dir = Path(b["results_dir"]) / ("smoke" if args.limit else "")
    raw_dir = results_dir / "benchmark"
    raw_dir.mkdir(parents=True, exist_ok=True)
    keys = runs_for(args.device)
    if args.runs:
        wanted = [k.strip() for k in args.runs.split(",")]
        unknown = [k for k in wanted if k not in RUNS]
        if unknown:
            print(f"Lượt chạy không tồn tại: {unknown}. Có: {list(RUNS)}")
            return 1
        keys = [k for k in keys if k in wanted]

    docs = load_documents(cfg, args.limit)
    print(f"{len(docs)} tài liệu, warmup {b['warmup'] if args.warmup is None else args.warmup}, "
          f"generation {gen_kwargs}, lượt chạy: {keys}")
    results: dict[str, dict] = {}
    for key in keys:
        reason = precheck(key, cfg, model_path)
        if reason:
            print(f"[{key}] bỏ qua — {reason}")
            results[key] = {"run": key, "status": "skipped", "reason": reason}
            continue
        out_json = raw_dir / f"{key}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", args.config, "--worker", key,
               "--out", str(out_json), "--model_path", model_path, "--log-level", args.log_level]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.length:
            cmd += ["--length", args.length]
        if args.warmup is not None:
            cmd += ["--warmup", str(args.warmup)]
        print(f"[{key}] đang chạy ...", flush=True)
        t = time.perf_counter()
        if out_json.exists():
            out_json.unlink()  # kết quả cũ của chính lượt chạy này
        proc = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if out_json.exists():
            results[key] = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            results[key] = {"run": key, "status": "error", "reason": f"worker thoát mã {proc.returncode}, không ghi kết quả"}
        r = results[key]
        if r.get("status") == "ok":
            import numpy as np

            print(f"[{key}] xong sau {time.perf_counter() - t:.0f} giây: p50 {np.percentile(r['latency_ms'], 50):.0f} ms, "
                  f"p95 {np.percentile(r['latency_ms'], 95):.0f} ms, input khớp phase 3 {r['input_matches_phase3']}/{r['n']}")
        else:
            print(f"[{key}] LỖI — {r.get('reason')}")
            if r.get("traceback"):
                print(r["traceback"])

    import pandas as pd

    pred = pd.DataFrame({"id": docs["id"], "dataset": docs["dataset"], "reference": docs["summary"]})
    for key, r in results.items():
        if r.get("status") == "ok":
            pred[key] = pd.Series(r["predictions"], index=pred.index)
    pred.to_csv(raw_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    info = system_info()
    (raw_dir / "system_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    command = "python scripts/benchmark.py " + " ".join(sys.argv[1:] if argv is None else argv)
    report = results_dir / "benchmark.md"
    warnings = write_report(results, docs["summary"].tolist(), cfg, info, report, command=command, gen_kwargs=gen_kwargs)
    for w in warnings:
        print(f"CẢNH BÁO: {w}")
    print(f"\nĐã ghi: {report}, {raw_dir / 'predictions.csv'}, {raw_dir}/*.json")
    return 0 if any(r.get("status") == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
