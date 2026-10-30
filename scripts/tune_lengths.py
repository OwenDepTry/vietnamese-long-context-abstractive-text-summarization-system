#!/usr/bin/env python
"""Bước 2 — tune length_penalty / min_new_tokens của mức `medium` trên tập VALIDATION.

Thử lưới nhỏ tham số trên cùng các mẫu validation (seed), chấm ROUGE so với reference, chọn cấu hình
ROUGE-L cao nhất mà không làm tăng lặp quá ngưỡng -> results/tune_lengths.md (+ .csv). Không tự sửa config.

    # thử nhanh (vài phút):
    python scripts/tune_lengths.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged --n 40
    # đầy đủ (GPU khuyến nghị):
    python scripts/tune_lengths.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged

Bản tóm tắt của từng ứng viên được lưu ngay vào results/tune_lengths/<ứng viên>.jsonl, nên chạy lại sau
khi bị ngắt sẽ bỏ qua phần đã xong.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import generation_kwargs, load_inference_config  # noqa: E402
from vnsum.inference.tune import (  # noqa: E402
    candidate_grid, candidate_name, gen_kwargs_for, pick_best, render_report, summarize_scores,
)


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--n", type=int, default=300, help="tổng số mẫu validation, chia đều cho các dataset")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--length", default="medium", help="mức độ dài cần tune (giữ max_new_tokens của mức này)")
    p.add_argument("--length_penalties", default="1.0,1.5,2.0")
    p.add_argument("--min_new_tokens", default="0,16,24,32")
    p.add_argument("--max_repetition_increase", type=float, default=1.0,
                   help="loại ứng viên có lặp 3-gram cao hơn baseline quá số điểm này")
    p.add_argument("--out", default="results/tune_lengths.md")
    args = p.parse_args(argv)

    import pandas as pd

    from vnsum.eval.config import load_eval_config
    from vnsum.eval.dataset import load_eval_set
    from vnsum.eval.metrics import repetition_rate, rouge_per_sample, syllables
    from vnsum.inference.predictor import build_backend

    cfg = load_inference_config(args.config)
    ecfg = copy.deepcopy(load_eval_config(cfg["benchmark"]["eval_config"]))
    ecfg["eval_set"]["split"] = "validation"
    ecfg["eval_set"]["samples_per_dataset"] = max(1, args.n // len(ecfg["eval_set"]["datasets"]))
    df = load_eval_set(ecfg)
    texts, refs = df["input_text"].fillna("").tolist(), df["summary"].fillna("").tolist()

    base = generation_kwargs(cfg, args.length)
    cands = candidate_grid(_floats(args.length_penalties), _ints(args.min_new_tokens))
    backend = build_backend(cfg, "pytorch", model_path=args.model_path, device=args.device)
    print(f"{len(df)} mẫu validation × {len(cands)} ứng viên, backend {backend.name} {backend.dtype} {backend.device}, "
          f"batch {args.batch_size}")

    out = Path(args.out)
    cache_dir = out.with_suffix("")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for k, cand in enumerate(cands, 1):
        name = candidate_name(cand)
        cache = cache_dir / f"{name}.jsonl"
        done = {}
        if cache.exists():
            done = {r["id"]: r["summary"] for r in map(json.loads, cache.read_text(encoding="utf-8").splitlines())}
        todo = [i for i, x in enumerate(df["id"].astype(str)) if x not in done]
        gk = gen_kwargs_for(base, cand)
        with cache.open("a", encoding="utf-8") as f:
            for s in range(0, len(todo), args.batch_size):
                idx = todo[s:s + args.batch_size]
                preds = backend.generate([texts[i] for i in idx], gk).texts
                for i, pred in zip(idx, preds):
                    sid = str(df["id"].iloc[i])
                    done[sid] = pred
                    f.write(json.dumps({"id": sid, "summary": pred}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{k}/{len(cands)}] {name}: {len(done)}/{len(df)}", end="\r", flush=True)
        preds = [done[str(x)] for x in df["id"]]
        scores = rouge_per_sample(preds, refs)
        frames.append(pd.DataFrame({
            "candidate": name, "id": df["id"].astype(str), "dataset": df["dataset"],
            "rouge1": [s["rouge1"] for s in scores], "rouge2": [s["rouge2"] for s in scores],
            "rougeL": [s["rougeL"] for s in scores],
            "length_syllables": [len(syllables(x)) for x in preds],
            "repetition_3gram": [repetition_rate(x, 3) for x in preds],
        }))
        print(f"  [{k}/{len(cands)}] {name}: ROUGE-L {frames[-1]['rougeL'].mean():.2f}, "
              f"{frames[-1]['length_syllables'].mean():.1f} âm tiết" + " " * 20)

    rows = pd.concat(frames, ignore_index=True)
    table = summarize_scores(rows)
    baseline = candidate_name(cands[0])
    best = pick_best(table, baseline, args.max_repetition_increase)
    best_cand = next(c for c in cands if candidate_name(c) == best)
    header = [
        "# Bước 2 — tune độ dài trên tập validation", "",
        f"Sinh tự động lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `scripts/tune_lengths.py`. "
        f"{len(df)} mẫu **validation** ({', '.join(f'{d}: {c}' for d, c in df['dataset'].value_counts().items())}), "
        f"input `input_text` như lúc train, backend {backend.name} {backend.dtype} {backend.device}. "
        f"Mức `{args.length}`: num_beams={base['num_beams']}, no_repeat_ngram_size={base['no_repeat_ngram_size']}, "
        f"max_new_tokens={base['max_new_tokens']}. Ứng viên bị loại nếu lặp 3-gram cao hơn baseline quá "
        f"{args.max_repetition_increase:g} điểm.",
    ]
    report = render_report(table, best, baseline, header=header,
                           ref_syllables=sum(len(syllables(r)) for r in refs) / len(refs),
                           best_cand=best_cand, max_new_tokens=int(base["max_new_tokens"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    rows.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8")
    print("\n" + report)
    print(f"Đã ghi: {out}, {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
