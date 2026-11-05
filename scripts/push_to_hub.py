#!/usr/bin/env python
"""Đưa model đã merge (+ bản ONNX INT8, tùy chọn) lên Hugging Face Hub, kèm model card sinh từ results/.

Mặc định CHỈ CHẠY THỬ: ghi model card ra results/model_card.md và liệt kê file sẽ upload, không gửi gì. Thêm
--push để upload thật (cần đăng nhập: `huggingface-cli login`, hoặc biến môi trường HF_TOKEN). Repo mới tạo ở
chế độ PRIVATE; thêm --public khi đã kiểm tra giấy phép dữ liệu (xem model card).

    python scripts/push_to_hub.py --repo_id <tên-tài-khoản>/vit5-vnsum --model_dir C:/vnsum-runs/vit5-vnsum-merged \\
        --onnx_dir C:/vnsum-runs/vit5-vnsum-merged-onnx/int8
    # đọc results/model_card.md, rồi chạy lại cùng lệnh với --push
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.model_card import build_model_card, load_metrics  # noqa: E402

# Chỉ những file cần để nạp model; bỏ checkpoint, optimizer, log, file tạm.
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.model"]
ONNX_PATTERNS = ["*.onnx", "*.onnx_data", "*.json", "*.model"]


def plan(folder: Path, patterns: list[str]) -> list[Path]:
    files = sorted({f for p in patterns for f in folder.glob(p) if f.is_file()})
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True, help="vd. ten-tai-khoan/vit5-vnsum")
    p.add_argument("--model_dir", required=True, help="thư mục model đã merge (scripts/merge_lora.py)")
    p.add_argument("--onnx_dir", default=None, help="thư mục ONNX INT8 (tùy chọn), upload vào onnx/int8/")
    p.add_argument("--metrics", default="results/metrics.csv")
    p.add_argument("--card_out", default="results/model_card.md")
    p.add_argument("--push", action="store_true", help="upload thật (mặc định chỉ chạy thử)")
    p.add_argument("--public", action="store_true", help="tạo repo công khai (mặc định private)")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None
    model_files = plan(model_dir, MODEL_PATTERNS)
    if not any(f.suffix == ".safetensors" for f in model_files):
        raise SystemExit(f"Không thấy *.safetensors trong {model_dir}: cần thư mục model ĐÃ MERGE.")
    onnx_files = plan(onnx_dir, ONNX_PATTERNS) if onnx_dir else []
    if onnx_dir and not any(f.suffix == ".onnx" for f in onnx_files):
        raise SystemExit(f"Không thấy *.onnx trong {onnx_dir}")

    card = build_model_card(load_metrics(args.metrics), args.repo_id, has_onnx=bool(onnx_files))
    Path(args.card_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.card_out).write_text(card, encoding="utf-8")

    total = sum(f.stat().st_size for f in model_files + onnx_files) / 1024 ** 2
    print(f"Repo: {args.repo_id} ({'public' if args.public else 'private'})")
    for f in model_files:
        print(f"  {f.name:40s} {f.stat().st_size / 1024 ** 2:9.1f} MB")
    for f in onnx_files:
        print(f"  onnx/int8/{f.name:30s} {f.stat().st_size / 1024 ** 2:9.1f} MB")
    print(f"Tổng: {total:,.0f} MB. Model card: {args.card_out}")
    if not args.push:
        print("CHƯA upload (chạy thử). Đọc model card, rồi chạy lại với --push.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=str(model_dir), allow_patterns=MODEL_PATTERNS,
                      commit_message="Model đã merge (ViT5-base + LoRA)")
    if onnx_dir:
        api.upload_folder(repo_id=args.repo_id, folder_path=str(onnx_dir), path_in_repo="onnx/int8",
                          allow_patterns=ONNX_PATTERNS, commit_message="ONNX INT8 (CPU)")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo_id,
                    commit_message="Model card")
    print(f"Đã upload: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
