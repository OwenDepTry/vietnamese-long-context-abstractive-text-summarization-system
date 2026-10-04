# vnsum — Vietnamese Long-Context Abstractive Summarization

Tóm tắt trừu tượng văn bản tiếng Việt dài với ViT5 (`VietAI/vit5-base`). Huấn
luyện trên GPU T4 16GB (Colab/Kaggle, **không bf16**); máy local chỉ dùng để
viết code và chạy smoke test trên CPU.

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 1 | Data pipeline: tải, chuẩn hóa, lọc, xử lý văn bản dài | ✅ |
| 2 | Training | ⏳ |
| 3 | Evaluation | ⏳ |
| 4 | Inference optimization | ⏳ |
| 5 | API / UI / Docker | ⏳ |

## Cấu trúc

```
configs/data.yaml          # mọi đường dẫn + hyperparameter của phase 1
scripts/prepare_data.py    # chạy toàn pipeline, ghi Parquet + thống kê
src/vnsum/
  config.py                # đọc + kiểm tra YAML
  data/
    load.py                # VietNews, WikiLingua(vi) -> {id, document, summary}
    preprocess.py          # NFC, dấu thanh, HTML, ký tự điều khiển, tách câu, lọc, dedup, tách từ (tùy chọn)
    long_context.py        # extractive_filter (BM25/LexRank) | hierarchical chunking
    tokenization.py        # đếm/cắt theo token ViT5
  models/ eval/ inference/ api/   # phase 2–5
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

## Giấy phép dữ liệu

WikiLingua: CC BY 3.0 (nội dung wikiHow). VietNews: xem bài báo VNDS (Nguyen et al., NICS 2019) và repo gốc `ThanhChinhBK/vietnews`.
