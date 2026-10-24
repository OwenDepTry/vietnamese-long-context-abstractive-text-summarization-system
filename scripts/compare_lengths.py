#!/usr/bin/env python
"""So sánh ba mức độ dài short / medium / long trên cùng tài liệu test -> results/lengths.md.

Đo cho từng mức: số token sinh (TB / trung vị), độ dài (âm tiết), latency, ROUGE-L so với reference,
tỷ lệ trigram lặp; và tỷ lệ tài liệu mà hai mức cho ra bản tóm tắt giống hệt nhau.

    python scripts/compare_lengths.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
    # thêm văn bản của riêng bạn (vd. văn bản quy định đã thử trên UI):
    python scripts/compare_lengths.py --config configs/inference.yaml --model_path ... --text_file vanban.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.inference.config import generation_kwargs, load_inference_config  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--backend", choices=["pytorch", "onnx"], default=None)
    p.add_argument("--variant", choices=["fp32", "int8"], default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--n", type=int, default=20, help="số tài liệu test")
    p.add_argument("--text_file", default=None, help="thêm một văn bản riêng vào cuối report")
    p.add_argument("--out", default="results/lengths.md")
    args = p.parse_args(argv)

    from vnsum.eval.config import load_eval_config
    from vnsum.eval.dataset import load_eval_set
    from vnsum.eval.metrics import repetition_rate, rouge_per_sample, syllables
    from vnsum.inference.bench import select_documents
    from vnsum.inference.predictor import Summarizer

    cfg = load_inference_config(args.config)
    lengths = list(cfg["lengths"])
    summ = Summarizer.from_config(cfg, args.backend, model_path=args.model_path, variant=args.variant, device=args.device)
    b = summ.backend
    df = load_eval_set(load_eval_config(cfg["benchmark"]["eval_config"]))
    docs = select_documents(df, args.n, int(cfg.get("seed", 42)))
    strategy = cfg["benchmark"]["strategy"]
    print(f"{len(docs)} tài liệu × {len(lengths)} mức, backend {b.name} {b.dtype} {b.device}")

    summ.summarize(docs["document"].iloc[0], length=lengths[0], strategy=strategy)  # warmup
    res: dict[str, list] = {k: [] for k in lengths}
    for i, row in enumerate(docs.itertuples(), 1):
        for k in lengths:
            res[k].append(summ.summarize(row.document, length=k, strategy=strategy))
        print(f"  {i}/{len(docs)}", end="\r", flush=True)
    refs = docs["summary"].tolist()

    lines = [
        "# So sánh mức độ dài short / medium / long", "",
        f"Sinh tự động lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `scripts/compare_lengths.py`. "
        f"{len(docs)} tài liệu test cố định (seed {cfg.get('seed', 42)}), backend {b.name} {b.dtype} {b.device}, "
        f"chiến lược `{strategy}`; latency là sau 1 lần warmup.", "",
        "| Mức | Tham số | Token sinh TB (trung vị) | Âm tiết TB | Latency TB (ms) | ROUGE-L | Lặp 3-gram (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in lengths:
        r = res[k]
        toks = [x.new_tokens for x in r]
        g = {kk: v for kk, v in generation_kwargs(cfg, k).items() if kk not in ("num_beams", "no_repeat_ngram_size", "early_stopping")}
        rl = statistics.mean(s["rougeL"] for s in rouge_per_sample([x.summary for x in r], refs))
        rep = statistics.mean(repetition_rate(x.summary, 3) for x in r)  # repetition_rate đã ở thang 0–100
        lines.append(f"| `{k}` | {', '.join(f'{kk}={v}' for kk, v in g.items())} | {statistics.mean(toks):.1f} ({statistics.median(toks):.0f}) | "
                     f"{statistics.mean(len(syllables(x.summary)) for x in r):.1f} | "
                     f"{statistics.mean(x.timings_ms['total_ms'] for x in r):,.0f} | {rl:.2f} | {rep:.2f} |")
    lines += ["", "## Hai mức cho cùng kết quả", "", "| Cặp | Số tài liệu giống hệt |", "|---|---|"]
    for a, c in combinations(lengths, 2):
        same = sum(x.summary == y.summary for x, y in zip(res[a], res[c]))
        lines.append(f"| `{a}` = `{c}` | {same}/{len(docs)} |")
    lines += ["", "## Ví dụ (3 tài liệu đầu)"]
    for i in range(min(3, len(docs))):
        lines += ["", f"### {docs['id'].iloc[i]}", "", f"**Reference:** {refs[i]}", ""]
        lines += [f"- **{k}** ({res[k][i].new_tokens} token): {res[k][i].summary}" for k in lengths]
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
        lines += ["", f"## Văn bản riêng: `{Path(args.text_file).name}`", ""]
        for k in lengths:
            r = summ.summarize(text, length=k)
            lines.append(f"- **{k}** ({r.new_tokens} token, {r.timings_ms['total_ms']:,.0f} ms, chiến lược {r.strategy}): {r.summary}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[4:4 + 2 + len(lengths)]))
    print(f"\nĐã ghi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
