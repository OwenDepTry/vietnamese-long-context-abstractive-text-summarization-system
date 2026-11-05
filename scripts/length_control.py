#!/usr/bin/env python
"""Đối chứng độ dài: hệ thống A thắng B vì nội dung tốt hơn, hay chỉ vì viết dài hơn?

ROUGE F1 có lợi cho bản tóm tắt dài gần bằng reference, nên so hai hệ thống có độ dài khác nhau dễ kết luận
sai. Script cắt từng dự đoán của A về đúng độ dài (âm tiết) của dự đoán B trên CÙNG mẫu, chấm lại ROUGE và
so cặp (paired bootstrap). Chỉ cắt khi A dài hơn B; mẫu nào A ngắn hơn thì giữ nguyên (A không được lợi).
Không cần model hay GPU: chỉ đọc ``per_sample.csv`` của ``scripts/evaluate.py``.

    python scripts/length_control.py --per_sample results/long_v2/per_sample.csv \\
        --system vit5_lora_hier_concat --baselines vit5_lora,vit5_lora_tuned

Ghi ``length_control.md`` cạnh ``per_sample.csv``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vnsum.data.preprocess import split_sentences  # noqa: E402
from vnsum.eval.metrics import rouge_per_sample, syllables  # noqa: E402
from vnsum.eval.stats import paired_bootstrap  # noqa: E402

METRICS = ("rouge1", "rouge2", "rougeL")


def n_syllables(text: str) -> int:
    return len(syllables(text or ""))


def cut_syllables(text: str, n: int) -> str:
    """Giữ các từ đầu (theo khoảng trắng) cho tới khi đủ ``n`` âm tiết; có thể cắt giữa câu."""
    out, count = [], 0
    for tok in (text or "").split():
        k = n_syllables(tok)
        if count + k > n and out:
            break
        out.append(tok)
        count += k
    return " ".join(out)


def cut_sentences(text: str, n: int) -> str:
    """Giữ các câu đầu sao cho tổng độ dài gần ``n`` âm tiết nhất (luôn giữ ít nhất một câu)."""
    sents = split_sentences(text or "")
    if not sents:
        return ""
    best, acc = sents[0], []
    for s in sents:
        acc.append(s)
        cand = " ".join(acc)
        if abs(n_syllables(cand) - n) <= abs(n_syllables(best) - n):
            best = cand
        else:
            break
    return best


def matched(preds: list[str], target_lengths: list[int], mode: str) -> list[str]:
    cut = cut_syllables if mode == "syllables" else cut_sentences
    return [cut(p, n) if n_syllables(p) > n else p for p, n in zip(preds, target_lengths)]


def score(preds: list[str], refs: list[str]) -> dict[str, np.ndarray]:
    rows = rouge_per_sample(preds, refs)
    return {m: np.array([r[m] for r in rows]) for m in METRICS}


def _fmt(r: dict[str, float]) -> str:
    p = "< 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
    return f"{r['diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}], p {p}"


def build_report(df: pd.DataFrame, system: str, baselines: list[str], *, n_boot: int, seed: int) -> str:
    by = {s: g.set_index("id") for s, g in df.groupby("system")}
    for s in [system, *baselines]:
        if s not in by:
            raise SystemExit(f"Không có hệ thống {s} trong per_sample.csv (có: {sorted(by)})")
    ids = by[system].index
    refs = by[system].loc[ids, "summary"].fillna("").tolist()
    sys_preds = by[system].loc[ids, "prediction"].fillna("").tolist()
    sys_scores = score(sys_preds, refs)

    lines = [
        "# Đối chứng độ dài",
        "",
        f"Sinh tự động lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `scripts/length_control.py`, "
        f"{len(ids)} mẫu. Dự đoán của `{system}` được cắt về đúng độ dài (âm tiết) của dự đoán đối chứng trên "
        "cùng mẫu (chỉ cắt khi dài hơn), chấm lại ROUGE, rồi so cặp: hiệu = "
        f"`{system}` − đối chứng, CI 95% paired bootstrap ({n_boot:,} lần).",
        "",
    ]
    for base in baselines:
        b_preds = by[base].loc[ids, "prediction"].fillna("").tolist()
        b_scores = score(b_preds, refs)
        target = [n_syllables(p) for p in b_preds]
        variants = {
            f"`{system}` nguyên bản": sys_preds,
            f"`{system}` cắt theo âm tiết": matched(sys_preds, target, "syllables"),
            f"`{system}` cắt theo câu": matched(sys_preds, target, "sentences"),
        }
        lines += [
            f"## So với `{base}` (độ dài TB {np.mean(target):.1f} âm tiết)",
            "",
            "| Biến thể | Âm tiết TB | ROUGE-L | Hiệu ROUGE-L | Hiệu ROUGE-2 |",
            "|---|---|---|---|---|",
            f"| `{base}` (đối chứng) | {np.mean(target):.1f} | {b_scores['rougeL'].mean():.2f} | | |",
        ]
        for name, preds in variants.items():
            sc = sys_scores if preds is sys_preds else score(preds, refs)
            rl = paired_bootstrap(sc["rougeL"], b_scores["rougeL"], n_boot, 0.95, seed)
            r2 = paired_bootstrap(sc["rouge2"], b_scores["rouge2"], n_boot, 0.95, seed)
            lines.append(f"| {name} | {np.mean([n_syllables(p) for p in preds]):.1f} | {sc['rougeL'].mean():.2f} | "
                         f"{_fmt(rl)} | {_fmt(r2)} |")
        lines.append("")
    lines += [
        "Đọc: nếu hiệu chỉ có ý nghĩa ở bản nguyên bản mà mất khi cắt về cùng độ dài, phần thắng chủ yếu đến từ "
        "độ dài. Cắt theo âm tiết có thể cắt giữa câu (hơi bất lợi cho hệ thống bị cắt); cắt theo câu giữ câu "
        "trọn vẹn nhưng độ dài chỉ gần đúng.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_sample", required=True)
    p.add_argument("--system", required=True, help="hệ thống bị nghi thắng nhờ độ dài")
    p.add_argument("--baselines", required=True, help="các hệ thống đối chứng, cách nhau bằng dấu phẩy")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    path = Path(args.per_sample)
    report = build_report(pd.read_csv(path), args.system, [b.strip() for b in args.baselines.split(",")],
                          n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out) if args.out else path.with_name("length_control.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Đã ghi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
