# vnsum — Vietnamese Long-Context Abstractive Summarization

Tóm tắt trừu tượng văn bản tiếng Việt dài với ViT5 (`VietAI/vit5-base`). Huấn
luyện trên GPU T4 16GB (Colab/Kaggle, **không bf16**); máy local chỉ dùng để
viết code và chạy smoke test trên CPU.

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 1 | Data pipeline: tải, chuẩn hóa, lọc, xử lý văn bản dài | ✅ |
| 2 | Training: ViT5 + LoRA (peft), Seq2SeqTrainer | ✅ code (chờ train T4) |
| 3 | Evaluation | ⏳ |
| 4 | Inference optimization | ⏳ |
| 5 | API / UI / Docker | ⏳ |

## Cấu trúc

```
configs/data.yaml          # mọi đường dẫn + hyperparameter của phase 1
scripts/prepare_data.py    # chạy toàn pipeline, ghi Parquet + thống kê
scripts/merge_lora.py      # merge adapter vào ViT5, lưu model đầy đủ + kiểm tra
configs/train.yaml         # hyperparameter phase 2 (đề xuất cho T4 16GB)
configs/train_rtx4050.yaml # override cho RTX 4050 Laptop 6GB (bf16)
notebooks/train_colab.ipynb  # Colab: chỉ gọi các script trên
src/vnsum/
  config.py                # đọc + kiểm tra YAML
  data/
    load.py                # VietNews, WikiLingua(vi) -> {id, document, summary}
    preprocess.py          # NFC, dấu thanh, HTML, ký tự điều khiển, tách câu, lọc, dedup, tách từ (tùy chọn)
    long_context.py        # extractive_filter (BM25/LexRank) | hierarchical chunking
    tokenization.py        # đếm/cắt theo token ViT5
  models/
    train.py               # Seq2SeqTrainer + LoRA, cấu hình từ configs/train.yaml
    train_config.py        # đọc/kiểm tra config, profile smoke, override CLI
    data.py                # đọc Parquet phase 1 -> dataset đã tokenize
    lora.py                # liệt kê module, chọn target_modules, đếm tham số
    callbacks.py           # guard loss NaN/inf + ROUGE tập con val mỗi epoch
    rouge.py               # ROUGE tokenizer tiếng Việt (giữ dấu)
  eval/ inference/ api/    # phase 3–5
tests/                     # pytest (offline, tokenizer giả) + test tokenizer ViT5 thật (tự skip nếu offline)
notebooks/  ui/
```

## Dữ liệu

| Dataset | Hub ID | Split | Ghi chú |
|---|---|---|---|
| VietNews (VNDS) | [`nam194/vietnews`](https://huggingface.co/datasets/nam194/vietnews) @ `39e2837` | train 99,134 / val 22,184 / test 22,498 | Văn bản trên Hub đã tách từ (`Hà_Nội`) và tách dấu câu (` , `); pipeline khôi phục văn bản tự nhiên (`desegment: true`). |
| WikiLingua (vi) | [`esdurmus/wiki_lingua`](https://huggingface.co/datasets/esdurmus/wiki_lingua), config `vietnamese` | chỉ có train (6,616 bài, nhiều section/bài) | Tự chia train/val/test 90/5/5 theo `hash(seed, url)`, nên các section của cùng một bài luôn chung một split. |

Schema đầu ra chung: `id, document, summary`, cùng các cột độ dài token và
các cột của chiến lược xử lý văn bản dài.

## Cài đặt & chạy

```bash
make install                      # pip install -r requirements.txt && pip install -e .
export HF_TOKEN=...               # tùy chọn; token chỉ đọc từ biến môi trường
make test                         # pytest
make smoke                        # = python scripts/prepare_data.py --config configs/data.yaml --limit 200
make prepare                      # chạy đầy đủ
```

`--limit N` là tổng số mẫu thô của cả lần chạy, chia đều cho các cặp
(dataset, split) và đọc theo kiểu streaming. Đầu ra gồm
`data/processed/<dataset>/<split>.parquet` và `data/processed/stats.json`.

## Thiết kế chính (phase 1)

- **Chuẩn hóa:** bỏ HTML → Unicode NFC → bỏ ký tự điều khiển/zero-width → (VietNews) khôi phục văn bản bị tách từ → thống nhất dấu thanh cho vần `oa/oe/uy` mở (`tone_style: old` cho ra `hòa, khỏe, thủy`) → gộp khoảng trắng.
- **Tách từ** (underthesea/pyvi) chỉ chạy khi bật `preprocess.word_segmentation.enabled` và ghi ra cột riêng `*_ws` (cho PhoBERT). Input của ViT5 không bao giờ bị tách từ.
- **Lọc:** document hoặc summary rỗng/quá ngắn, summary dài hơn document, trùng lặp trong split và giữa các split (ưu tiên giữ ở test > validation > train).
- **Văn bản dài** (`long_context.strategy`). Mọi độ dài đều đo bằng tokenizer ViT5 và đã trừ chỗ cho `</s>`:
  - `extractive_filter`: chấm điểm câu bằng BM25-centrality hoặc LexRank, chọn câu rồi xuất theo thứ tự gốc, luôn ≤ `max_input_tokens`. Hai chế độ chọn câu (`selection`):
    - `budget` (mặc định): nạp câu theo thứ hạng tới khi đầy ngân sách token.
    - `ratio`: giữ top 30% số câu (baseline).

    Chạy đầy đủ VietNews train với `ratio` cho thấy document bị lọc (trung vị 1,170 token) chỉ còn khoảng 515 token, tức bỏ phí khoảng một nửa ngữ cảnh; vì vậy mặc định được chuyển sang `budget`.
  - `hierarchical`: gom câu thành chunk ≤ `chunk_max_tokens`, overlap khoảng `overlap_tokens` token tính theo câu. Câu nào dài hơn cả một chunk mới bị chia, theo dấu phẩy trước rồi mới tới token.

## Phase 2 — Training

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

## Giấy phép dữ liệu

WikiLingua: CC BY 3.0 (nội dung wikiHow). VietNews: xem bài báo VNDS (Nguyen et al., NICS 2019) và repo gốc `ThanhChinhBK/vietnews`.
