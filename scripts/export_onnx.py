#!/usr/bin/env python
"""Phase 4: export model đã merge sang ONNX (có KV cache) + dynamic INT8, rồi kiểm tra bằng một lần sinh.

Chạy trong .venv-onnx:
    python scripts/export_onnx.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
    # thêm bản FP16 cho GPU (chạy cần onnxruntime-gpu):
    python scripts/export_onnx.py --config configs/inference.yaml --model_path ... --fp16

Thư mục đã có file .onnx thì bỏ qua bước export/quantize (chỉ kiểm tra lại); dùng --force để export lại.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import load_inference_config, onnx_dirs, resolve_model_path  # noqa: E402
from vnsum.inference.hardware import ONNX_WEIGHT_PATTERNS, PYTORCH_WEIGHT_PATTERNS, dir_size_mb  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--fp16", action="store_true", help="tạo thêm bản FP16 (chuyển từ bản FP32) cho GPU")
    p.add_argument("--skip_int8", action="store_true")
    p.add_argument("--force", action="store_true", help="export/quantize lại MỌI bản dù thư mục đích đã có file .onnx")
    p.add_argument("--rebuild", default="", help="chỉ làm lại các bản này, vd. --rebuild fp16 (ghi đè file cũ)")
    p.add_argument("--text_file", default=None, help="văn bản dùng để kiểm tra (mặc định: đoạn mẫu có sẵn)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _has_onnx(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.onnx"))


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_inference_config(args.config)
    model_path = resolve_model_path(cfg, args.model_path)
    if not Path(model_path).is_dir():
        print(f"Không thấy model đã merge ở {model_path}; truyền --model_path")
        return 1
    dirs = onnx_dirs(cfg, model_path)

    from vnsum.inference.export import (
        SAMPLE_TEXT, check_vietnamese_output, convert_fp16, ensure_tokenizer, export_onnx, quantize_int8,
    )
    from vnsum.inference.predictor import OnnxBackend, PyTorchBackend, Summarizer, onnx_session_kwargs

    steps: list[tuple[str, float]] = []
    rebuild = {v.strip() for v in args.rebuild.split(",") if v.strip()}
    if args.force or "fp32" in rebuild or not _has_onnx(dirs["fp32"]):
        t = time.perf_counter()
        export_onnx(model_path, dirs["fp32"], cfg)
        steps.append(("export fp32", time.perf_counter() - t))
    else:
        print(f"[fp32] đã có {dirs['fp32']}, bỏ qua export (dùng --force để làm lại)")
    if ensure_tokenizer(model_path, dirs["fp32"]):
        print(f"[fp32] đã lưu tokenizer vào {dirs['fp32']}")
    if not args.skip_int8:
        if args.force or "int8" in rebuild or not _has_onnx(dirs["int8"]):
            t = time.perf_counter()
            quantize_int8(dirs["fp32"], dirs["int8"], cfg)
            steps.append(("quantize int8", time.perf_counter() - t))
        else:
            print(f"[int8] đã có {dirs['int8']}, bỏ qua quantize")
    if args.fp16:
        if args.force or "fp16" in rebuild or not _has_onnx(dirs["fp16"]):
            t = time.perf_counter()
            convert_fp16(dirs["fp32"], dirs["fp16"])
            steps.append(("convert fp16", time.perf_counter() - t))
        else:
            print(f"[fp16] đã có {dirs['fp16']}, bỏ qua export")

    # Lần chạy trước có thể đã export xong nhưng dừng trước khi lưu tokenizer -> bổ sung, không export lại.
    for v in ("fp32", "int8", "fp16"):
        if _has_onnx(dirs[v]) and ensure_tokenizer(model_path, dirs[v]):
            print(f"[{v}] đã lưu tokenizer vào {dirs[v]}")

    for name, sec in steps:
        print(f"{name}: {sec:.1f} giây")

    # ----- kiểm tra: kích thước + một lần sinh trên từng bản
    from vnsum.config import load_config

    data_cfg = load_config(cfg["data_config"])
    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else SAMPLE_TEXT
    print(f"\nKích thước trọng số: PyTorch {dir_size_mb(model_path, PYTORCH_WEIGHT_PATTERNS):,.0f} MB", end="")
    variants = [v for v in ("fp32", "int8", "fp16") if _has_onnx(dirs[v])]
    for v in variants:
        print(f" | ONNX {v} {dir_size_mb(dirs[v], ONNX_WEIGHT_PATTERNS):,.0f} MB", end="")
    print("\n")

    failed = False
    reference: str | None = None
    backends = [("pytorch_cpu", lambda: PyTorchBackend(model_path, "cpu", "fp32", int(cfg["max_input_tokens"])))]
    for v in variants:
        dev = "cuda" if v == "fp16" else "cpu"
        backends.append((f"onnx_{v}_{dev}", lambda v=v, dev=dev: OnnxBackend(dirs[v], v, dev, **onnx_session_kwargs(cfg), max_input_tokens=int(cfg["max_input_tokens"]))))
    for name, make in backends:
        try:
            summ = Summarizer(make(), cfg, data_cfg)
            res = summ.summarize(text, length="medium")
        except Exception as err:  # in lỗi từng bản, không dừng các bản còn lại
            print(f"[{name}] LỖI: {type(err).__name__}: {err}")
            failed = True
            continue
        problems = check_vietnamese_output(res.summary)
        if name == "pytorch_cpu":
            reference = res.summary
        status = "OK" if not problems else "KHÔNG HỢP LỆ: " + "; ".join(problems)
        if reference is not None and name != "pytorch_cpu":
            status += " | giống hệt PyTorch" if res.summary == reference else " | KHÁC PyTorch"
        print(f"[{name}] {res.timings_ms['total_ms']:.0f} ms (lần chạy đầu, chưa warmup), {res.new_tokens} token — {status}")
        print(f"    {res.summary}")
        failed |= bool(problems) and name != "pytorch_cpu"
    print("\nSố ms ở trên là lần chạy đầu (gồm khởi tạo), không dùng làm số benchmark; xem scripts/benchmark.py.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
