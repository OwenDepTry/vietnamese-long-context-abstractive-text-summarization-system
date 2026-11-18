# Phase 2 — Training

← [Về README](../README.md)

```bash
# Local (CPU) — cài torch CPU trước: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m vnsum.models.train --config configs/train.yaml --max_steps 5 --limit 32      # smoke test
python scripts/merge_lora.py --config configs/train.yaml --adapter outputs/vit5-lora/final_adapter
```

Trên Colab dùng `notebooks/train_colab.ipynb` (T4): checkpoint lưu trên Google Drive, chạy lại cell train sẽ tự resume.

- **LoRA:** r=16, alpha=32, gắn vào `q,k,v,o` của mọi attention (self-attn encoder, self + cross-attn decoder). Bật `lora.include_ffn` để thêm `wi, wo`.
- **Precision:** `configs/train.yaml` (T4) mặc định fp32, vì T4 không có bf16 và T5 dễ NaN ở fp16. `--precision fp16` nhanh hơn; nếu loss NaN/inf, guard dừng ngay và báo lỗi. GPU hỗ trợ bf16 (Ampere/Ada) dùng `precision: bf16`, ví dụ `configs/train_rtx4050.yaml`; config này kế thừa `train.yaml` qua `extends`.
- **Smoke test đã chạy thật (RTX 4050 Laptop):** LoRA trên `q,k,v,o` cho 3,538,944 tham số trainable / tổng 229,489,920 (1.542%). Sau khi merge, logits lệch so với PeftModel tối đa 2.29e-05.
- **Input:** `data.input_strategy` = `extractive_filter` (cột `input_text` của phase 1) hoặc `truncate` (baseline). `hierarchical` dùng lúc suy luận, không dùng để train.
- **ROUGE** trong lúc train tính trên 500 mẫu val ở cuối mỗi epoch, với tokenizer giữ dấu tiếng Việt (tokenizer mặc định của `rouge_score` xóa mọi chữ có dấu).

## Kết quả train (RTX 4050 Laptop 6GB, `configs/train_rtx4050.yaml`)

| | Giá trị |
|---|---|
| Dữ liệu | 116,557 mẫu train (VietNews 99,133 + WikiLingua-vi 17,424), input `extractive_filter[bm25, budget]` |
| Cấu hình | LoRA r=16 trên `q,k,v,o`; bf16; batch 2 × grad accum 16 (batch hiệu dụng 32); LR 5e-4; 1 epoch = 3,643 step |
| Thời gian | 7 giờ 26 phút (~7.35 s/step), VRAM ~2.2 GB |
| Train loss | 4.43 → ~1.85 (trung bình cả epoch 1.938) |
| Eval loss (500 mẫu val) | 2.61 (sau 20 step, lần đo tốc độ) → **1.899** (cuối epoch) |
| ROUGE-1/2/L (500 mẫu val, greedy, tokenizer âm tiết giữ dấu) | 14.88/6.52/12.40 (sau 20 step) → **29.88 / 15.72 / 25.35** (cuối epoch) |
| Merge | max \|Δlogits\| giữa PeftModel và model đã merge = 1.14e-05 |

ROUGE ở bảng trên là số đo sơ bộ trong lúc train: greedy decoding, không chặn lặp từ, chỉ 500 mẫu val. Đánh giá đầy đủ trên tập test ở phase 3 (beam search, `no_repeat_ngram_size`, tách riêng nhóm document > 1024 token).

Ví dụ đầu ra của model đã merge (một bài trong tập test VietNews):
> Sau khi lừa đảo vợ chồng ông Đinh Ngọc H., Huy đã lừa vợ chồng ông Th. 250 triệu đồng.
