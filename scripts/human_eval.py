#!/usr/bin/env python
"""Đối chiếu LLM-as-Judge với điểm người chấm (không gọi API).

    # 1) Tạo bộ chấm ẩn danh từ kết quả evaluate.py (cần results/per_sample.csv + data/processed)
    python scripts/human_eval.py export --config configs/eval.yaml
    # 2) Đọc results/human_eval/items.md, điền human_score (1–5) vào results/human_eval/annotation_sheet.csv
    # 3) Tính mức đồng thuận -> results/human_eval/agreement.md
    python scripts/human_eval.py analyze --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from vnsum.eval.config import load_eval_config  # noqa: E402
from vnsum.eval.human_eval import analyze, export_items, read_sheet, select_items  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["export", "analyze"])
    p.add_argument("--config", required=True)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--processed_dir", default=None)
    p.add_argument("--force", action="store_true", help="export: ghi đè annotation_sheet.csv đã có (MẤT điểm đã chấm)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_eval_config(args.config)
    if args.processed_dir:
        cfg["paths"]["processed_dir"] = args.processed_dir
    results_dir = Path(args.results_dir or cfg["paths"]["results_dir"])
    hcfg = cfg.get("human_eval") or {}
    out_dir = results_dir / hcfg.get("dir", "human_eval")

    if args.command == "export":
        from vnsum.eval.dataset import load_eval_set

        per_sample_path = results_dir / "per_sample.csv"
        if not per_sample_path.exists():
            print(f"Không thấy {per_sample_path}. Chạy scripts/evaluate.py (có judge) trước.")
            return 1
        per_sample = pd.read_csv(per_sample_path, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=True)
        per_sample["prediction"] = per_sample["prediction"].fillna("")
        items = select_items(per_sample, list(hcfg.get("systems", ["vit5_lora", "vit5_public", "llm_zeroshot"])),
                             int(hcfg.get("n_items", 50)), int(hcfg.get("seed", 42)))
        eval_df = load_eval_set(cfg)
        docs = eval_df.assign(id=eval_df["id"].astype(str)).set_index("id")["document"]
        try:
            paths = export_items(items, docs, out_dir, int(cfg["judge"]["max_document_chars"]), force=args.force)
        except FileExistsError as err:
            print(err)
            return 1
        print(f"Đã tạo {len(items)} mục ({items['system'].value_counts().to_dict()}):")
        print(f"  Đọc:      {paths['items']}")
        print(f"  Điền điểm: {paths['sheet']}  (cột human_score, 1–5)")
        print(f"  Đáp án:   {paths['key']}  (KHÔNG mở trước khi chấm xong)")
        return 0

    sheet_path, key_path = out_dir / "annotation_sheet.csv", out_dir / "key.csv"
    for p in (sheet_path, key_path):
        if not p.exists():
            print(f"Không thấy {p}. Chạy 'export' trước.")
            return 1
    sheet = read_sheet(sheet_path)
    key = pd.read_csv(key_path, encoding="utf-8-sig", dtype={"item_id": str, "id": str})
    text, merged = analyze(sheet, key, cfg)
    (out_dir / "agreement.md").write_text(text, encoding="utf-8")
    merged.to_csv(out_dir / "merged.csv", index=False, encoding="utf-8-sig")
    print(text)
    print(f"Đã ghi: {out_dir / 'agreement.md'}, {out_dir / 'merged.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
