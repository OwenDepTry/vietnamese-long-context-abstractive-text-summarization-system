# Phase 3 — Báo cáo đánh giá

Sinh tự động lúc 2026-11-05 13:12 UTC bởi `scripts/evaluate.py`. Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.

## Thiết lập

- Tập đánh giá: split `test`, vietnews: 1,000 mẫu, wikilingua_vi: 910 mẫu; nhóm long (document > 1024 token): 225 mẫu.
- Generation (model seq2seq): num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True.
- Hệ thống: `vit5_lora` = ViT5-base + LoRA r=16 (phase 2), C:/vnsum-runs/vit5-vnsum-merged (input `input_text`); `vit5_distill` = ViT5 + LoRA, train tiếp trên bản tóm tắt của LLM (bước B), C:/vnsum-runs/vit5-vnsum-distill-merged (input `input_text`).
- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).
- BERTScore: vinai/phobert-base, num_layers=10, tách từ underthesea, ≤ 256 subword, **raw (không rescale)**, thang 0–100.
- BERTScore: 16/7,640 chuỗi (prediction + reference) bị cắt ở 256 subword PhoBERT.

**Không có trong kết quả:** `vit5_lora` (judge) (cần --allow_api); `vit5_distill` (judge) (cần --allow_api).

## Kết quả chính (toàn bộ tập đánh giá)

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) |
|---|---|---|---|---|---|
| `vit5_lora` | 1,910 | 33.46 | 18.42 | 26.98 | 58.68 |
| `vit5_distill` | 1,910 | 31.04 | 15.99 | 24.33 | 55.90 |

## Theo dataset

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora` | vietnews | 1,000 | 33.75 | 19.28 | 26.75 | 56.59 |
| `vit5_lora` | wikilingua_vi | 910 | 33.15 | 17.47 | 27.23 | 60.97 |
| `vit5_distill` | vietnews | 1,000 | 31.72 | 17.65 | 24.71 | 55.16 |
| `vit5_distill` | wikilingua_vi | 910 | 30.29 | 14.17 | 23.91 | 56.71 |

## Theo độ dài document (long vs short)

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora` | long | 225 | 28.01 | 13.73 | 20.70 | 55.34 |
| `vit5_lora` | short | 1,685 | 34.19 | 19.05 | 27.82 | 59.12 |
| `vit5_distill` | long | 225 | 26.83 | 12.30 | 19.86 | 53.36 |
| `vit5_distill` | short | 1,685 | 31.60 | 16.48 | 24.92 | 56.24 |

## Chỉ số phụ (toàn bộ tập đánh giá)

Lặp = tỷ lệ n-gram trong summary là bản lặp của n-gram đứng trước. Novel = tỷ lệ n-gram của summary không có trong document gốc.

|  | Độ dài TB (âm tiết) | repetition 3gram (%) | novel 1gram (%) | novel 2gram (%) |
|---|---|---|---|---|
| `vit5_lora` | 26.8 | 0.12 | 2.21 | 17.84 |
| `vit5_distill` | 35.2 | 0.15 | 1.84 | 21.37 |
| *reference (tham chiếu)* | 39.2 | 1.20 | 18.08 | 54.20 |

## Khoảng tin cậy 95% (bootstrap, 1,000 lần lấy lại)

Định dạng: trung bình [cận dưới, cận trên]. Faithfulness tính trên các mẫu được judge chấm.

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|
| `vit5_lora` | 33.46 [32.76, 34.12] | 18.42 [17.78, 18.98] | 26.98 [26.30, 27.57] | 58.68 [58.22, 59.09] |
| `vit5_distill` | 31.04 [30.46, 31.63] | 15.99 [15.46, 16.52] | 24.33 [23.77, 24.91] | 55.90 [55.49, 56.29] |

## So sánh cặp với `vit5_lora` (paired bootstrap)

Hiệu = vit5_lora − hệ thống, tính trên cùng các mẫu. Khác biệt được coi là có ý nghĩa khi CI 95% của hiệu không chứa 0. Chưa hiệu chỉnh cho nhiều phép so sánh (multiple comparisons), nên các p sát ngưỡng cần đọc thận trọng.

| So với | Chỉ số | n | Hiệu | CI 95% | p | Kết luận |
|---|---|---|---|---|---|---|
| `vit5_distill` | ROUGE-1 | 1,910 | +2.42 | [+1.86, +3.04] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_distill` | ROUGE-2 | 1,910 | +2.43 | [+1.91, +2.98] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_distill` | ROUGE-L | 1,910 | +2.65 | [+2.08, +3.21] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_distill` | BERTScore-F | 1,910 | +2.78 | [+2.43, +3.14] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |

## Giới hạn của phương pháp đánh giá

- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp 10, input tách từ bằng underthesea (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ 256 subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa 200 mẫu/hệ thống (ngẫu nhiên, seed 42) bằng một model (claude-sonnet-4-6); độ tin cậy của judge được đối chiếu với người chấm trong `human_eval/agreement.md` (nếu đã chạy); judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn 30000 ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ 30000 ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm vietnews 1,000 mẫu, wikilingua_vi 910 mẫu; nhóm "long" (document > 1024 token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. Khoảng tin cậy và so sánh cặp ở trên dùng bootstrap trên toàn bộ tập, chưa tách theo dataset hay độ dài; các bảng theo tập con chỉ là trung bình điểm.

