#!/usr/bin/env python
"""Đo và kiểm chứng bộ kiểm tra sự thật bề mặt (vnsum.factcheck) trên kết quả ĐÃ CÓ — không sinh lại, không GPU.

* % bản tóm tắt có số / ngày / tên riêng không có trong văn bản gốc, theo hệ thống (và của reference).
* Đối chiếu với điểm faithfulness của LLM judge (nếu per_sample.csv có): dữ liệu ĐỘC LẬP với 40 mục người chấm
  đã dùng để chỉnh quy tắc, nên đây mới là phép kiểm chứng khách quan.
* Đối chiếu với điểm người chấm (human_eval/merged.csv, nếu có) — chỉ để tham khảo (đã dùng để chỉnh quy tắc).

    python scripts/factcheck_report.py --config configs/eval.yaml --results_dir results
    python scripts/factcheck_report.py --config configs/eval.yaml --results_dir results/long_v2

Ghi <results_dir>/factcheck.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _p(x: float) -> str:
    return f"{x:.1f}" if x == x else "—"


def build_report(per_sample, documents, human=None) -> str:
    import pandas as pd

    from vnsum.eval.metrics import unsupported_in

    df = per_sample.copy()
    df["id"] = df["id"].astype(str)
    df["document"] = df["id"].map(documents).fillna("")
    df = df[df["document"] != ""]
    df["items"] = [unsupported_in(str(p) if pd.notna(p) else "", d) for p, d in zip(df["prediction"], df["document"])]
    df["flag"] = df["items"].map(bool)
    ref = df.drop_duplicates("id")
    ref_flag = 100 * pd.Series([bool(unsupported_in(str(s), d)) for s, d in zip(ref["summary"], ref["document"])]).mean()

    lines = ["# Kiểm tra sự thật bề mặt (số / ngày / tên riêng)", "",
             "Sinh bởi `scripts/factcheck_report.py` từ dự đoán đã có. Một bản tóm tắt bị **đánh dấu** khi có số, ngày "
             "hoặc tên riêng không tìm thấy trong văn bản gốc (`src/vnsum/factcheck.py`). Chỉ bắt lỗi bề mặt; KHÔNG bắt "
             "lỗi quan hệ (gán nhầm lời nói, đảo ý, trộn vai trò).", "",
             "## Tỷ lệ bị đánh dấu", "", "| Hệ thống | n | Bị đánh dấu (%) |", "|---|---:|---:|"]
    for system, g in df.groupby("system", sort=False):
        lines.append(f"| `{system}` | {len(g):,} | {_p(100 * g['flag'].mean())} |")
    lines.append(f"| *reference* | {len(ref):,} | {_p(ref_flag)} |")
    lines += ["", "Reference cũng có thể bị đánh dấu: bản tóm tắt gốc của bài báo đôi khi nêu số / tên không có trong "
              "thân bài, nên con số này là mức nền chứ không phải 0."]

    if "faithfulness" in df and df["faithfulness"].notna().any():
        j = df.dropna(subset=["faithfulness"])
        lines += ["", "## Đối chiếu với điểm faithfulness của LLM judge (dữ liệu độc lập)", "",
                  "| | n | Faithfulness TB | Judge ≤ 2 (%) |", "|---|---:|---:|---:|"]
        for flag, label in ((True, "Bị đánh dấu"), (False, "Không bị đánh dấu")):
            g = j[j["flag"] == flag]
            lines.append(f"| {label} | {len(g):,} | {g['faithfulness'].mean():.2f} | {_p(100 * (g['faithfulness'] <= 2).mean())} |")
        low = j["faithfulness"] <= 2
        recall = 100 * j.loc[low, "flag"].mean() if low.any() else float("nan")
        precision = 100 * (j.loc[j["flag"], "faithfulness"] <= 2).mean() if j["flag"].any() else float("nan")
        lines += ["", f"Trong các bản judge chấm ≤ 2, bị đánh dấu: **{_p(recall)}%** (độ phủ). Trong các bản bị đánh dấu, "
                  f"judge chấm ≤ 2: **{_p(precision)}%** (độ chính xác). Judge có thể sai; đây là đối chiếu, không phải sự thật."]

    if human is not None and len(human):
        h = human.copy()
        h["id"] = h["id"].astype(str)
        h = h.merge(df[["system", "id", "flag"]], on=["system", "id"], how="inner")
        if len(h):
            lines += ["", "## Đối chiếu với người chấm (tham khảo — đã dùng để chỉnh quy tắc)", "",
                      "| | n | Trung thực TB (người) |", "|---|---:|---:|"]
            for flag, label in ((True, "Bị đánh dấu"), (False, "Không bị đánh dấu")):
                g = h[h["flag"] == flag]
                lines.append(f"| {label} | {len(g)} | {g['human_score'].mean():.2f} |")

    ex = df[df["flag"]].head(10)
    if len(ex):
        lines += ["", "## Ví dụ bị đánh dấu", "", "| Hệ thống | id | Không tìm thấy trong bài |", "|---|---|---|"]
        for _, r in ex.iterrows():
            lines.append(f"| `{r['system']}` | {r['id']} | {'; '.join(r['items']).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--results_dir", default="results")
    args = p.parse_args(argv)

    import pandas as pd

    from vnsum.eval.config import load_eval_config
    from vnsum.eval.dataset import load_eval_set

    results = Path(args.results_dir)
    per_sample = pd.read_csv(results / "per_sample.csv", encoding="utf-8-sig", dtype={"id": str})
    eval_df = load_eval_set(load_eval_config(args.config))
    documents = dict(zip(eval_df["id"].astype(str), eval_df["document"].fillna("")))
    merged = results / "human_eval" / "merged.csv"
    human = pd.read_csv(merged, encoding="utf-8-sig", dtype={"id": str}) if merged.exists() else None
    report = build_report(per_sample, documents, human)
    out = results / "factcheck.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Đã ghi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
