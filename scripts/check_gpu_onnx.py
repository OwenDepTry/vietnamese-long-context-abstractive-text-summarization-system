#!/usr/bin/env python
"""Chẩn đoán ONNX trên GPU: lỗi do FP16 hay do đường chạy CUDA (io_binding)?

Chạy cùng đoạn văn mẫu trên ONNX FP32 và FP16 với CUDA, bật/tắt io_binding, so với PyTorch CPU.

    python scripts/check_gpu_onnx.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import load_inference_config, onnx_dirs, resolve_model_path  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    args = p.parse_args(argv)

    from vnsum.config import load_config
    from vnsum.inference.export import SAMPLE_TEXT, check_vietnamese_output
    from vnsum.inference.predictor import OnnxBackend, PyTorchBackend, Summarizer

    cfg = load_inference_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    model_path = resolve_model_path(cfg, args.model_path)
    dirs = onnx_dirs(cfg, model_path)
    mx = int(cfg["max_input_tokens"])

    def run(name, make):
        try:
            b = make()
            r = Summarizer(b, cfg, data_cfg).summarize(SAMPLE_TEXT, length="medium")
        except Exception as err:
            print(f"[{name}] LỖI: {type(err).__name__}: {err}\n")
            return None
        problems = check_vietnamese_output(r.summary)
        print(f"[{name}] {r.new_tokens} token — {'OK' if not problems else 'KHÔNG HỢP LỆ: ' + '; '.join(problems)}")
        print(f"    providers: {getattr(b, 'providers', '-')}, io_binding: {getattr(b, 'session_settings', {}).get('use_io_binding', '-')}")
        print(f"    {r.summary}\n")
        del b
        gc.collect()
        return r.summary

    ref = run("pytorch_cpu", lambda: PyTorchBackend(model_path, "cpu", "fp32", mx))
    cases = [(v, io) for v in ("fp32", "fp16") for io in (True, False)]
    for variant, io in cases:
        if not any(dirs[variant].glob("*.onnx")):
            print(f"[onnx_{variant}_cuda] bỏ qua — chưa có {dirs[variant]}\n")
            continue
        out = run(f"onnx_{variant}_cuda io_binding={io}",
                  lambda v=variant, io=io: OnnxBackend(dirs[v], v, "cuda", None, mx, allow_spinning=False, use_io_binding=io))
        if out is not None and ref is not None:
            print(f"    -> {'giống hệt' if out == ref else 'KHÁC'} PyTorch CPU\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
