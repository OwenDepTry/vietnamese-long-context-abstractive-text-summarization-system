#!/usr/bin/env python
"""Tóm tắt một văn bản bằng Summarizer (backend PyTorch hoặc ONNX, chọn trong configs/inference.yaml).

    python scripts/summarize.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged --file bai.txt
    python scripts/summarize.py --config configs/inference.yaml --model_path ... --file bai.txt --length long --strategy hierarchical
    python scripts/summarize.py --config configs/inference.yaml --model_path ... --backend pytorch --text "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import load_inference_config  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="file văn bản UTF-8")
    src.add_argument("--text")
    p.add_argument("--backend", choices=["pytorch", "onnx"], default=None)
    p.add_argument("--variant", choices=["fp32", "int8", "fp16"], default=None, help="bản ONNX")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--length", default=None, help="short | medium | long (theo mục lengths)")
    p.add_argument("--strategy", choices=["auto", "extractive", "hierarchical"], default=None)
    args = p.parse_args(argv)

    cfg = load_inference_config(args.config)
    from vnsum.inference.predictor import Summarizer

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    summ = Summarizer.from_config(cfg, args.backend, model_path=args.model_path, variant=args.variant, device=args.device)
    r = summ.summarize(text, length=args.length, strategy=args.strategy)
    b = summ.backend
    print(f"[{b.name} {b.dtype} {b.device}] chiến lược {r.strategy}, input {r.input_tokens} token -> model {r.model_input_tokens} token, "
          f"{r.generate_calls} lần sinh, {r.timings_ms['total_ms']:.0f} ms (gồm lần chạy đầu)")
    if r.chunk_summaries:
        print(f"\nTóm tắt {len(r.chunk_summaries)} chunk (vòng 1):")
        for i, c in enumerate(r.chunk_summaries, 1):
            print(f"  {i}. {c}")
    print(f"\n{r.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
