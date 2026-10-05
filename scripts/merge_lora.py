#!/usr/bin/env python
"""Merge LoRA adapter vào ViT5 gốc, lưu model + tokenizer đầy đủ cho phase 4, rồi kiểm tra.

Kiểm tra sau khi merge:
  1. Logits của model đã merge khớp với PeftModel (trước khi merge) trên cùng input.
  2. Load lại model đã lưu từ đĩa và generate 1 câu tóm tắt.

Ví dụ:
    python scripts/merge_lora.py --config configs/train.yaml
    python scripts/merge_lora.py --config configs/train.yaml \\
        --adapter outputs/vit5-lora/checkpoint-5 --output outputs/vit5-vnsum-merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.models.train_config import load_train_config  # noqa: E402

_DEMO_TEXT = (
    "Ngày 5/10, Sở Giao thông vận tải TP. Hồ Chí Minh cho biết sẽ phân luồng giao thông trên nhiều tuyến "
    "đường ở quận 1 để phục vụ thi công nhà ga metro. Thời gian phân luồng kéo dài ba tháng, người dân được "
    "khuyến cáo đi theo các tuyến thay thế để tránh ùn tắc vào giờ cao điểm."
)


def _sample_text(cfg: dict) -> str:
    """Lấy 1 input thật từ tập test của phase 1 nếu có, nếu không dùng câu demo."""
    try:
        from vnsum.models.data import read_split

        name = cfg["data"]["datasets"][0]
        df = read_split(cfg["paths"]["processed_dir"], name, "test", cfg["data"]["input_strategy"])
        if len(df):
            return str(df["source"].iloc[0])
    except Exception:  # noqa: BLE001 - chỉ là nguồn câu thử
        pass
    return _DEMO_TEXT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None, help="thư mục adapter (mặc định <output_dir>/final_adapter)")
    p.add_argument("--output", default=None, help="thư mục lưu model đã merge (mặc định paths.merged_dir)")
    p.add_argument("--text", default=None, help="văn bản để thử generate (mặc định: 1 mẫu test của phase 1)")
    p.add_argument("--max_new_tokens", type=int, default=None, help="mặc định: data.max_target_length")
    args = p.parse_args(argv)

    cfg = load_train_config(args.config)
    adapter = Path(args.adapter or Path(cfg["paths"]["output_dir"]) / "final_adapter")
    output = Path(args.output or cfg["paths"]["merged_dir"])
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"Không thấy adapter_config.json trong {adapter}")

    import os

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSeq2SeqLM

    from vnsum.data.tokenization import load_tokenizer

    token = os.environ.get(cfg["hf"]["token_env"]) or None
    base_name = PeftConfig.from_pretrained(str(adapter)).base_model_name_or_path
    tok_source = str(adapter) if (adapter / "tokenizer.json").exists() else base_name
    tokenizer = load_tokenizer(tok_source, cfg["hf"]["token_env"])
    print(f"Base: {base_name}\nAdapter: {adapter}\nTokenizer: {tok_source}")

    base = AutoModelForSeq2SeqLM.from_pretrained(base_name, token=token, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    peft_model.eval()

    text = args.text or _sample_text(cfg)
    max_src = int(cfg["data"]["max_source_length"])
    enc = tokenizer(text, max_length=max_src, truncation=True, return_tensors="pt")
    dec = torch.tensor([[peft_model.config.decoder_start_token_id]])
    with torch.no_grad():
        logits_peft = peft_model(**enc, decoder_input_ids=dec).logits

    merged = peft_model.merge_and_unload()
    merged.eval()
    with torch.no_grad():
        logits_merged = merged(**enc, decoder_input_ids=dec).logits
    max_diff = float((logits_peft - logits_merged).abs().max())
    print(f"Max |logits PeftModel - merged| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit(f"Merge sai lệch quá lớn ({max_diff}); không lưu.")

    merged.config.use_cache = True
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "merge_info.json").write_text(
        json.dumps({"base_model": base_name, "adapter": str(adapter), "max_logit_diff": max_diff}, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu model + tokenizer: {output}")

    # Load lại từ đĩa như phase 4 sẽ làm, rồi generate.
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(str(output), dtype=torch.float32).eval()
    tok2 = load_tokenizer(str(output))
    enc2 = tok2(text, max_length=max_src, truncation=True, return_tensors="pt")
    max_new = args.max_new_tokens or int(cfg["data"]["max_target_length"])
    with torch.no_grad():
        out = reloaded.generate(**enc2, max_new_tokens=max_new, num_beams=1)
    summary = tok2.decode(out[0], skip_special_tokens=True)
    print(f"\nInput (300 ký tự đầu): {text[:300]}")
    print(f"Tóm tắt từ model đã merge: {summary!r}")
    print(f"Token sinh ra (kể cả special): {tok2.decode(out[0], skip_special_tokens=False)!r}")
    if not summary.strip():
        # Adapter mới train vài step (smoke test) có thể chỉ sinh token <extra_id_*> của
        # pretraining span-corruption -> rỗng sau khi bỏ special token. Không coi là lỗi merge.
        print("CẢNH BÁO: tóm tắt rỗng sau khi bỏ special token (bình thường với adapter smoke test).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
