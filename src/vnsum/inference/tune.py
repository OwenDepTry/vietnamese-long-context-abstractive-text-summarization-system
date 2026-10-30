"""Bước 2 — tune tham số độ dài (``length_penalty``, ``min_new_tokens``) trên tập VALIDATION.

Vấn đề (results/report.md): bản tóm tắt của ViT5 + LoRA trung bình 26.8 âm tiết, reference 39.2 —
quá ngắn làm ROUGE recall thấp. Thay vì đoán tham số, thử một lưới nhỏ trên split validation (không
dùng test, tránh chọn tham số theo tập test) rồi chọn cấu hình có ROUGE-L cao nhất mà không làm
tăng tỷ lệ lặp quá ``max_repetition_increase`` điểm so với baseline (tham số hiện tại).

Phần logic ở đây không cần torch; ``scripts/tune_lengths.py`` lo việc nạp model và sinh.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import pandas as pd

METRICS = ("rouge1", "rouge2", "rougeL", "length_syllables", "repetition_3gram")


def candidate_grid(length_penalties: Sequence[float], min_new_tokens: Sequence[int]) -> list[dict[str, Any]]:
    """Mọi tổ hợp (length_penalty, min_new_tokens), baseline (1.0, 0) luôn đứng đầu."""
    grid = [{"length_penalty": float(lp), "min_new_tokens": int(mn)} for lp, mn in product(length_penalties, min_new_tokens)]
    base = {"length_penalty": 1.0, "min_new_tokens": 0}
    rest = [c for c in grid if c != base]
    return [base] + sorted(rest, key=lambda c: (c["min_new_tokens"], c["length_penalty"]))


def candidate_name(c: dict[str, Any]) -> str:
    return f"lp{c['length_penalty']:g}_min{c['min_new_tokens']}"


def gen_kwargs_for(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """Generation kwargs = ``base`` (beam, no_repeat, max_new_tokens của mức đang tune) + tham số ứng viên.

    Bỏ các khóa mang giá trị mặc định để baseline sinh y hệt cấu hình hiện tại.
    """
    out = {k: v for k, v in base.items() if k not in ("length_penalty", "min_new_tokens")}
    if int(cand["min_new_tokens"]) > 0:
        if int(cand["min_new_tokens"]) >= int(out["max_new_tokens"]):
            raise ValueError("min_new_tokens phải nhỏ hơn max_new_tokens")
        out["min_new_tokens"] = int(cand["min_new_tokens"])
    if float(cand["length_penalty"]) != 1.0 and int(out.get("num_beams", 1)) > 1:
        out["length_penalty"] = float(cand["length_penalty"])
    return out


def summarize_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows``: một dòng mỗi (candidate, mẫu) với cột dataset + METRICS -> trung bình theo candidate,
    kèm cột riêng theo từng dataset cho ROUGE-L."""
    overall = rows.groupby("candidate", sort=False)[list(METRICS)].mean()
    overall["n"] = rows.groupby("candidate", sort=False).size()
    per_ds = rows.pivot_table(index="candidate", columns="dataset", values="rougeL", aggfunc="mean", sort=False)
    per_ds.columns = [f"rougeL_{c}" for c in per_ds.columns]
    return overall.join(per_ds)


def pick_best(table: pd.DataFrame, baseline: str, max_repetition_increase: float = 1.0) -> str:
    """Ứng viên có ROUGE-L cao nhất trong số không làm lặp tăng quá ngưỡng so với baseline."""
    limit = float(table.loc[baseline, "repetition_3gram"]) + float(max_repetition_increase)
    ok = table[table["repetition_3gram"] <= limit]
    return str(ok["rougeL"].idxmax())


def render_report(table: pd.DataFrame, best: str, baseline: str, *, header: Sequence[str],
                  ref_syllables: float, best_cand: dict[str, Any], max_new_tokens: int) -> str:
    ds_cols = [c for c in table.columns if c.startswith("rougeL_")]
    lines = list(header) + [
        "",
        f"Độ dài TB của reference: **{ref_syllables:.1f} âm tiết**.",
        "",
        "| Ứng viên | length_penalty | min_new_tokens | ROUGE-1 | ROUGE-2 | ROUGE-L | "
        + " | ".join(f"ROUGE-L {c[7:]}" for c in ds_cols) + " | Âm tiết TB | Lặp 3-gram (%) |",
        "|---" * (8 + len(ds_cols)) + "|",
    ]
    for name, r in table.iterrows():
        lp, mn = name.split("_min")
        mark = " **(chọn)**" if name == best else (" (hiện tại)" if name == baseline else "")
        lines.append(
            f"| `{name}`{mark} | {lp[2:]} | {mn} | {r.rouge1:.2f} | {r.rouge2:.2f} | {r.rougeL:.2f} | "
            + " | ".join(f"{r[c]:.2f}" for c in ds_cols)
            + f" | {r.length_syllables:.1f} | {r.repetition_3gram:.2f} |"
        )
    b, x = table.loc[baseline], table.loc[best]
    lines += [
        "",
        "## Kết luận",
        "",
        f"- Chọn `{best}`: ROUGE-L {x.rougeL:.2f} so với {b.rougeL:.2f} của cấu hình hiện tại "
        f"({x.rougeL - b.rougeL:+.2f}), độ dài {b.length_syllables:.1f} → {x.length_syllables:.1f} âm tiết.",
        "- Đây là số trên VALIDATION. Sau khi cập nhật config, chạy lại đánh giá trên test để có số cuối cùng.",
        "",
        "Bước tiếp theo: chép `min_new_tokens` / `length_penalty` dưới đây vào `configs/eval.yaml` "
        "(mục `systems.vit5_lora_tuned.generation`), đánh giá trên test; nếu vẫn tốt hơn thì cập nhật "
        "`configs/inference.yaml` (mục `lengths.medium`):",
        "",
        "```yaml",
        "medium:",
        f"  max_new_tokens: {max_new_tokens}",
    ]
    if best_cand["min_new_tokens"]:
        lines.append(f"  min_new_tokens: {best_cand['min_new_tokens']}")
    if best_cand["length_penalty"] != 1.0:
        lines.append(f"  length_penalty: {best_cand['length_penalty']:g}")
    lines += ["```", ""]
    return "\n".join(lines)
