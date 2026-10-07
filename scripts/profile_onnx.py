#!/usr/bin/env python
"""Chẩn đoán vì sao ONNX chậm trên CPU: thử vài cấu hình session của ONNX Runtime và tách thời gian
nằm trong ORT (encoder / decoder / decoder_with_past) với thời gian ngoài ORT (vòng generate của
transformers, chuyển tensor, sắp xếp lại KV cache cho beam search).

    python scripts/profile_onnx.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged

Đây là công cụ chẩn đoán (2 tài liệu); số liệu chính thức vẫn lấy từ scripts/benchmark.py.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import load_inference_config, onnx_dirs, resolve_model_path  # noqa: E402


def find_sessions(model) -> dict[str, object]:
    """Các InferenceSession bên trong ORTModelForSeq2SeqLM (encoder, decoder, decoder_with_past)."""
    found: dict[str, object] = {}
    for name, comp in vars(model).items():
        sess = getattr(comp, "session", None)
        if sess is not None and hasattr(sess, "run") and id(sess) not in {id(s) for s in found.values()}:
            found[name] = sess
    return found


def instrument(sessions: dict[str, object], timers: dict[str, list[float]]) -> None:
    for name, sess in sessions.items():
        for method in ("run", "run_with_iobinding"):
            orig = getattr(sess, method, None)
            if orig is None:
                continue

            def wrapper(*a, _orig=orig, _name=name, **k):
                t = time.perf_counter()
                try:
                    return _orig(*a, **k)
                finally:
                    timers[_name].append((time.perf_counter() - t) * 1000)

            setattr(sess, method, wrapper)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--variants", default="int8", help="fp32,int8")
    p.add_argument("--docs", type=int, default=2)
    p.add_argument("--skip_pytorch", action="store_true")
    args = p.parse_args(argv)

    import torch

    from vnsum.config import load_config
    from vnsum.eval.config import load_eval_config
    from vnsum.eval.dataset import load_eval_set
    from vnsum.inference.bench import select_documents
    from vnsum.inference.hardware import cpu_name
    from vnsum.inference.predictor import OnnxBackend, PyTorchBackend, Summarizer

    cfg = load_inference_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    model_path = resolve_model_path(cfg, args.model_path)
    eval_df = load_eval_set(load_eval_config(cfg["benchmark"]["eval_config"]))
    eval_df["id"] = eval_df["id"].astype(str)
    docs = select_documents(eval_df, args.docs, int(cfg.get("seed", 42)))["document"].tolist()
    strategy, length = cfg["benchmark"]["strategy"], cfg["benchmark"]["length"]
    n_threads = torch.get_num_threads()
    print(f"CPU: {cpu_name()} | luồng logic: {__import__('os').cpu_count()} | torch.get_num_threads(): {n_threads}")
    print(f"{len(docs)} tài liệu, length={length}, strategy={strategy}; mỗi cấu hình: 1 lần warmup rồi đo.\n")

    def measure(summ: Summarizer, timers=None):
        summ.summarize(docs[0], length=length, strategy=strategy)  # warmup
        if timers is not None:
            timers.clear()
        totals, steps = [], []
        for d in docs:
            r = summ.summarize(d, length=length, strategy=strategy)
            totals.append(r.timings_ms["total_ms"])
            steps.append(r.new_tokens)
        return totals, steps

    if not args.skip_pytorch:
        b = PyTorchBackend(model_path, "cpu", "fp32", int(cfg["max_input_tokens"]))
        totals, steps = measure(Summarizer(b, cfg, data_cfg))
        print(f"[pytorch fp32] ms/tài liệu: {[round(x) for x in totals]}  (token sinh: {steps})\n")
        del b
        gc.collect()

    settings = [
        ("mặc định ORT", {}),
        ("tắt spinning", {"allow_spinning": False}),
        (f"{n_threads} luồng + tắt spinning", {"intra_op_num_threads": n_threads, "allow_spinning": False}),
        ("io_binding + tắt spinning", {"use_io_binding": True, "allow_spinning": False}),
    ]
    dirs = onnx_dirs(cfg, model_path)
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for label, kw in settings:
            try:
                b = OnnxBackend(dirs[variant], variant, "cpu", max_input_tokens=int(cfg["max_input_tokens"]), **kw)
            except Exception as err:
                print(f"[onnx {variant} | {label}] LỖI khi nạp: {type(err).__name__}: {err}\n")
                continue
            timers: dict[str, list[float]] = defaultdict(list)
            sessions = find_sessions(b.model)
            if not sessions:
                print("    (không tìm thấy InferenceSession bên trong model; chỉ đo tổng thời gian)")
            instrument(sessions, timers)
            try:
                totals, steps = measure(Summarizer(b, cfg, data_cfg), timers)
            except Exception as err:
                print(f"[onnx {variant} | {label}] LỖI khi chạy: {type(err).__name__}: {err}\n")
                del b
                gc.collect()
                continue
            in_ort = sum(sum(v) for v in timers.values())
            print(f"[onnx {variant} | {label}] ms/tài liệu: {[round(x) for x in totals]}  (token sinh: {steps})")
            for name, v in timers.items():
                print(f"    {name:<20} {len(v):>5} lần gọi, tổng {sum(v):>9,.0f} ms, TB {sum(v) / max(len(v), 1):>7.1f} ms/lần")
            print(f"    trong ORT {in_ort:,.0f} ms / tổng {sum(totals):,.0f} ms -> ngoài ORT {sum(totals) - in_ort:,.0f} ms\n")
            del b
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
