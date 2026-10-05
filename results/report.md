# Phase 3 — Báo cáo đánh giá

Sinh tự động lúc 2026-10-05 14:02 UTC bởi `scripts/evaluate.py`. Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.

## Thiết lập

- Tập đánh giá: split `test`, vietnews: 1,000 mẫu, wikilingua_vi: 910 mẫu; nhóm long (document > 1024 token): 225 mẫu.
- Generation (model seq2seq): num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True.
- Hệ thống: `lead3` = 3 câu đầu của `document`; `vit5_lora` = ViT5-base + LoRA r=16 (phase 2), C:\vnsum-runs\vit5-vnsum-merged (input `input_text`); `vit5_public` = VietAI/vit5-base-vietnews-summarization (input `input_text`); `llm_zeroshot` = openai_compatible/gpt-6-luna zero-shot (input `document`).
- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).
- BERTScore: vinai/phobert-base, num_layers=10, tách từ underthesea, ≤ 256 subword, **raw (không rescale)**, thang 0–100.
- Faithfulness: LLM-as-Judge `claude-sonnet-4-6`, rubric 1–5, temperature 0, chấm trên cùng một mẫu ngẫu nhiên (seed 42): `lead3` 200 mẫu, `vit5_lora` 194 mẫu, `vit5_public` 195 mẫu, `llm_zeroshot` 200 mẫu.
- BERTScore: 33/15,280 chuỗi (prediction + reference) bị cắt ở 256 subword PhoBERT.

## Kết quả chính (toàn bộ tập đánh giá)

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) | Faithfulness (1–5) | n chấm |
|---|---|---|---|---|---|---|---|
| `lead3` | 1,910 | 24.91 | 11.41 | 17.17 | 51.15 | 5.00 | 200 |
| `vit5_lora` | 1,910 | 33.46 | 18.42 | 26.98 | 58.68 | 4.03 | 194 |
| `vit5_public` | 1,910 | 29.88 | 16.24 | 23.65 | 53.74 | 2.81 | 195 |
| `llm_zeroshot` | 1,910 | 30.69 | 13.28 | 21.37 | 53.07 | 4.92 | 200 |

## Theo dataset

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `lead3` | vietnews | 1,000 | 26.98 | 14.36 | 18.91 | 52.62 |
| `lead3` | wikilingua_vi | 910 | 22.63 | 8.16 | 15.24 | 49.53 |
| `vit5_lora` | vietnews | 1,000 | 33.75 | 19.28 | 26.75 | 56.59 |
| `vit5_lora` | wikilingua_vi | 910 | 33.15 | 17.47 | 27.23 | 60.97 |
| `vit5_public` | vietnews | 1,000 | 39.46 | 24.75 | 32.11 | 60.99 |
| `vit5_public` | wikilingua_vi | 910 | 19.36 | 6.89 | 14.35 | 45.78 |
| `llm_zeroshot` | vietnews | 1,000 | 30.78 | 14.68 | 21.06 | 53.59 |
| `llm_zeroshot` | wikilingua_vi | 910 | 30.60 | 11.75 | 21.70 | 52.50 |

## Theo độ dài document (long vs short)

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `lead3` | long | 225 | 22.27 | 8.48 | 14.38 | 48.60 |
| `lead3` | short | 1,685 | 25.26 | 11.80 | 17.54 | 51.49 |
| `vit5_lora` | long | 225 | 28.01 | 13.73 | 20.70 | 55.34 |
| `vit5_lora` | short | 1,685 | 34.19 | 19.05 | 27.82 | 59.12 |
| `vit5_public` | long | 225 | 23.81 | 11.20 | 17.87 | 49.46 |
| `vit5_public` | short | 1,685 | 30.69 | 16.91 | 24.42 | 54.31 |
| `llm_zeroshot` | long | 225 | 28.52 | 11.19 | 18.93 | 51.72 |
| `llm_zeroshot` | short | 1,685 | 30.98 | 13.56 | 21.69 | 53.25 |

## Chỉ số phụ (toàn bộ tập đánh giá)

Lặp = tỷ lệ n-gram trong summary là bản lặp của n-gram đứng trước. Novel = tỷ lệ n-gram của summary không có trong document gốc.

|  | Độ dài TB (âm tiết) | repetition 3gram (%) | novel 1gram (%) | novel 2gram (%) |
|---|---|---|---|---|
| `lead3` | 84.6 | 2.51 | 0.00 | 0.00 |
| `vit5_lora` | 26.8 | 0.12 | 2.21 | 17.84 |
| `vit5_public` | 29.4 | 0.04 | 12.92 | 41.18 |
| `llm_zeroshot` | 61.4 | 0.27 | 11.95 | 52.95 |
| *reference (tham chiếu)* | 39.2 | 1.20 | 18.08 | 54.20 |

## Phân bố điểm faithfulness

| Hệ thống | 1 | 2 | 3 | 4 | 5 | lỗi parse | TB |
|---|---|---|---|---|---|---|---|
| `lead3` | 0 | 0 | 0 | 0 | 200 | 0 | 5.00 |
| `vit5_lora` | 10 | 27 | 21 | 26 | 110 | 6 | 4.03 |
| `vit5_public` | 36 | 58 | 42 | 25 | 34 | 5 | 2.81 |
| `llm_zeroshot` | 0 | 0 | 0 | 15 | 185 | 0 | 4.92 |

## Giới hạn của phương pháp đánh giá

- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp 10, input tách từ bằng underthesea (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ 256 subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa 200 mẫu/hệ thống (ngẫu nhiên, seed 42) bằng một model (claude-sonnet-4-6), không có đối chứng người chấm; judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn 30000 ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ 30000 ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm vietnews 1,000 mẫu, wikilingua_vi 910 mẫu; nhóm "long" (document > 1024 token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. Report chưa có khoảng tin cậy hay kiểm định ý nghĩa thống kê.

