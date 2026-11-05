# Phase 3 — Báo cáo đánh giá

Sinh tự động lúc 2026-11-05 15:55 UTC bởi `scripts/evaluate.py`. Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.

## Thiết lập

- Tập đánh giá: split `test`, wikilingua_vi: 125 mẫu, vietnews: 100 mẫu; nhóm long (document > 1024 token): 225 mẫu.
- Generation (model seq2seq): num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True.
- Hệ thống: `vit5_lora` = ViT5-base + LoRA r=16 (phase 2), C:/vnsum-runs/vit5-vnsum-merged (input `input_text`); `vit5_lora_hier128` = ViT5 + LoRA, văn bản dài: tóm tắt phân cấp, chunk 128 token, C:/vnsum-runs/vit5-vnsum-merged (input `input_text`); `vit5_lora_hier_concat` = ViT5 + LoRA, văn bản dài: tóm tắt từng chunk, ghép + bỏ câu trùng (không sinh lại), C:/vnsum-runs/vit5-vnsum-merged (input `input_text`); `vit5_lora_hier_concat_b1` = ViT5 + LoRA, văn bản dài: concat, sinh chunk batch 1 (đối chứng batching), C:/vnsum-runs/vit5-vnsum-merged (input `input_text`); `vit5_lora_tuned` = ViT5 + LoRA, extractive ép viết dài (min_new_tokens 64) — đối chứng độ dài, C:/vnsum-runs/vit5-vnsum-merged (input `input_text`).
- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).
- BERTScore: vinai/phobert-base, num_layers=10, tách từ underthesea, ≤ 256 subword, **raw (không rescale)**, thang 0–100.
- BERTScore: 5/2,250 chuỗi (prediction + reference) bị cắt ở 256 subword PhoBERT.

**Không có trong kết quả:** `vit5_lora` (judge) (cần --allow_api); `vit5_lora_hier_concat` (judge) (cần --allow_api); `vit5_lora_hier_concat_b1` (judge) (cần --allow_api).

## Kết quả chính (toàn bộ tập đánh giá)

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) |
|---|---|---|---|---|---|
| `vit5_lora` | 225 | 28.01 | 13.73 | 20.70 | 55.34 |
| `vit5_lora_hier128` | 225 | 16.28 | 7.71 | 13.35 | 46.69 |
| `vit5_lora_hier_concat` | 225 | 30.90 | 15.04 | 22.29 | 57.08 |
| `vit5_lora_hier_concat_b1` | 225 | 30.86 | 14.97 | 22.29 | 57.05 |
| `vit5_lora_tuned` | 225 | 30.56 | 13.91 | 20.55 | 56.53 |

## Theo dataset

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora` | vietnews | 100 | 26.26 | 12.73 | 18.66 | 50.44 |
| `vit5_lora` | wikilingua_vi | 125 | 29.41 | 14.53 | 22.33 | 59.25 |
| `vit5_lora_hier128` | vietnews | 100 | 22.93 | 10.86 | 17.68 | 48.63 |
| `vit5_lora_hier128` | wikilingua_vi | 125 | 10.97 | 5.18 | 9.89 | 45.13 |
| `vit5_lora_hier_concat` | vietnews | 100 | 27.84 | 13.70 | 19.49 | 51.65 |
| `vit5_lora_hier_concat` | wikilingua_vi | 125 | 33.35 | 16.12 | 24.53 | 61.43 |
| `vit5_lora_hier_concat_b1` | vietnews | 100 | 27.65 | 13.51 | 19.40 | 51.63 |
| `vit5_lora_hier_concat_b1` | wikilingua_vi | 125 | 33.42 | 16.14 | 24.60 | 61.39 |
| `vit5_lora_tuned` | vietnews | 100 | 27.25 | 12.33 | 18.35 | 50.73 |
| `vit5_lora_tuned` | wikilingua_vi | 125 | 33.20 | 15.16 | 22.31 | 61.16 |

## Theo độ dài document (long vs short)

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora` | long | 225 | 28.01 | 13.73 | 20.70 | 55.34 |
| `vit5_lora_hier128` | long | 225 | 16.28 | 7.71 | 13.35 | 46.69 |
| `vit5_lora_hier_concat` | long | 225 | 30.90 | 15.04 | 22.29 | 57.08 |
| `vit5_lora_hier_concat_b1` | long | 225 | 30.86 | 14.97 | 22.29 | 57.05 |
| `vit5_lora_tuned` | long | 225 | 30.56 | 13.91 | 20.55 | 56.53 |

## Chỉ số phụ (toàn bộ tập đánh giá)

Lặp = tỷ lệ n-gram trong summary là bản lặp của n-gram đứng trước. Novel = tỷ lệ n-gram của summary không có trong document gốc.

|  | Độ dài TB (âm tiết) | repetition 3gram (%) | novel 1gram (%) | novel 2gram (%) |
|---|---|---|---|---|
| `vit5_lora` | 30.8 | 0.16 | 0.92 | 15.67 |
| `vit5_lora_hier128` | 16.0 | 0.02 | 2.55 | 18.56 |
| `vit5_lora_hier_concat` | 54.0 | 1.86 | 1.59 | 18.81 |
| `vit5_lora_hier_concat_b1` | 53.9 | 1.78 | 1.56 | 18.68 |
| `vit5_lora_tuned` | 60.7 | 0.33 | 2.15 | 21.03 |
| *reference (tham chiếu)* | 52.8 | 1.42 | 12.00 | 52.15 |

## Khoảng tin cậy 95% (bootstrap, 1,000 lần lấy lại)

Định dạng: trung bình [cận dưới, cận trên]. Faithfulness tính trên các mẫu được judge chấm.

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|
| `vit5_lora` | 28.01 [26.40, 29.59] | 13.73 [12.40, 15.10] | 20.70 [19.38, 22.11] | 55.34 [54.19, 56.45] |
| `vit5_lora_hier128` | 16.28 [14.83, 17.93] | 7.71 [6.53, 8.96] | 13.35 [12.01, 14.77] | 46.69 [45.64, 47.77] |
| `vit5_lora_hier_concat` | 30.90 [29.36, 32.39] | 15.04 [13.79, 16.40] | 22.29 [20.89, 23.68] | 57.08 [55.98, 58.18] |
| `vit5_lora_hier_concat_b1` | 30.86 [29.30, 32.32] | 14.97 [13.66, 16.29] | 22.29 [20.88, 23.67] | 57.05 [55.95, 58.15] |
| `vit5_lora_tuned` | 30.56 [29.15, 31.95] | 13.91 [12.81, 15.03] | 20.55 [19.48, 21.60] | 56.53 [55.43, 57.51] |

## So sánh cặp với `vit5_lora` (paired bootstrap)

Hiệu = vit5_lora − hệ thống, tính trên cùng các mẫu. Khác biệt được coi là có ý nghĩa khi CI 95% của hiệu không chứa 0. Chưa hiệu chỉnh cho nhiều phép so sánh (multiple comparisons), nên các p sát ngưỡng cần đọc thận trọng.

| So với | Chỉ số | n | Hiệu | CI 95% | p | Kết luận |
|---|---|---|---|---|---|---|
| `vit5_lora_hier128` | ROUGE-1 | 225 | +11.72 | [+9.93, +13.40] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier128` | ROUGE-2 | 225 | +6.02 | [+4.63, +7.27] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier128` | ROUGE-L | 225 | +7.35 | [+5.87, +8.72] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier128` | BERTScore-F | 225 | +8.65 | [+7.36, +9.92] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier_concat` | ROUGE-1 | 225 | -2.90 | [-4.24, -1.70] | < 0.001 | có ý nghĩa: vit5_lora_hier_concat cao hơn |
| `vit5_lora_hier_concat` | ROUGE-2 | 225 | -1.31 | [-2.43, -0.29] | 0.012 | có ý nghĩa: vit5_lora_hier_concat cao hơn |
| `vit5_lora_hier_concat` | ROUGE-L | 225 | -1.59 | [-2.72, -0.53] | 0.002 | có ý nghĩa: vit5_lora_hier_concat cao hơn |
| `vit5_lora_hier_concat` | BERTScore-F | 225 | -1.74 | [-2.56, -0.99] | < 0.001 | có ý nghĩa: vit5_lora_hier_concat cao hơn |
| `vit5_lora_hier_concat_b1` | ROUGE-1 | 225 | -2.85 | [-4.26, -1.57] | < 0.001 | có ý nghĩa: vit5_lora_hier_concat_b1 cao hơn |
| `vit5_lora_hier_concat_b1` | ROUGE-2 | 225 | -1.24 | [-2.41, -0.12] | 0.024 | có ý nghĩa: vit5_lora_hier_concat_b1 cao hơn |
| `vit5_lora_hier_concat_b1` | ROUGE-L | 225 | -1.59 | [-2.77, -0.44] | 0.002 | có ý nghĩa: vit5_lora_hier_concat_b1 cao hơn |
| `vit5_lora_hier_concat_b1` | BERTScore-F | 225 | -1.71 | [-2.58, -0.94] | < 0.001 | có ý nghĩa: vit5_lora_hier_concat_b1 cao hơn |
| `vit5_lora_tuned` | ROUGE-1 | 225 | -2.55 | [-3.64, -1.43] | < 0.001 | có ý nghĩa: vit5_lora_tuned cao hơn |
| `vit5_lora_tuned` | ROUGE-2 | 225 | -0.18 | [-0.96, +0.65] | 0.646 | chưa đủ bằng chứng khác biệt |
| `vit5_lora_tuned` | ROUGE-L | 225 | +0.15 | [-0.68, +1.04] | 0.716 | chưa đủ bằng chứng khác biệt |
| `vit5_lora_tuned` | BERTScore-F | 225 | -1.19 | [-1.79, -0.61] | < 0.001 | có ý nghĩa: vit5_lora_tuned cao hơn |

## Giới hạn của phương pháp đánh giá

- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp 10, input tách từ bằng underthesea (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ 256 subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa 200 mẫu/hệ thống (ngẫu nhiên, seed 42) bằng một model (claude-sonnet-4-6); độ tin cậy của judge được đối chiếu với người chấm trong `human_eval/agreement.md` (nếu đã chạy); judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn 30000 ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ 30000 ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm wikilingua_vi 125 mẫu, vietnews 100 mẫu; nhóm "long" (document > 1024 token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. Khoảng tin cậy và so sánh cặp ở trên dùng bootstrap trên toàn bộ tập, chưa tách theo dataset hay độ dài; các bảng theo tập con chỉ là trung bình điểm.

