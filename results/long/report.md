# Phase 3 — Báo cáo đánh giá

Sinh tự động lúc 2026-10-23 10:28 UTC bởi `scripts/evaluate.py`. Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.

## Thiết lập

- Tập đánh giá: split `test`, wikilingua_vi: 125 mẫu, vietnews: 100 mẫu; nhóm long (document > 1024 token): 225 mẫu.
- Generation (model seq2seq): num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True.
- Hệ thống: `vit5_lora` = ViT5-base + LoRA r=16 (phase 2), C:\vnsum-runs\vit5-vnsum-merged (input `input_text`); `vit5_lora_hier` = ViT5 + LoRA, văn bản dài: tóm tắt phân cấp, C:\vnsum-runs\vit5-vnsum-merged (input `input_text`).
- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).
- BERTScore: vinai/phobert-base, num_layers=10, tách từ underthesea, ≤ 256 subword, **raw (không rescale)**, thang 0–100.
- BERTScore: 2/900 chuỗi (prediction + reference) bị cắt ở 256 subword PhoBERT.

**Không có trong kết quả:** `vit5_lora` (judge) (cần --allow_api); `vit5_lora_hier` (judge) (cần --allow_api).

## Kết quả chính (toàn bộ tập đánh giá)

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) |
|---|---|---|---|---|---|
| `vit5_lora` | 225 | 28.01 | 13.73 | 20.70 | 55.34 |
| `vit5_lora_hier` | 225 | 16.30 | 7.91 | 13.48 | 46.77 |

## Theo dataset

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora` | vietnews | 100 | 26.26 | 12.73 | 18.66 | 50.44 |
| `vit5_lora` | wikilingua_vi | 125 | 29.41 | 14.53 | 22.33 | 59.25 |
| `vit5_lora_hier` | vietnews | 100 | 23.03 | 11.40 | 17.99 | 48.91 |
| `vit5_lora_hier` | wikilingua_vi | 125 | 10.92 | 5.12 | 9.87 | 45.07 |

## Theo độ dài document (long vs short)

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora` | long | 225 | 28.01 | 13.73 | 20.70 | 55.34 |
| `vit5_lora_hier` | long | 225 | 16.30 | 7.91 | 13.48 | 46.77 |

## Chỉ số phụ (toàn bộ tập đánh giá)

Lặp = tỷ lệ n-gram trong summary là bản lặp của n-gram đứng trước. Novel = tỷ lệ n-gram của summary không có trong document gốc.

|  | Độ dài TB (âm tiết) | repetition 3gram (%) | novel 1gram (%) | novel 2gram (%) |
|---|---|---|---|---|
| `vit5_lora` | 30.8 | 0.16 | 0.92 | 15.67 |
| `vit5_lora_hier` | 15.9 | 0.02 | 2.53 | 18.55 |
| *reference (tham chiếu)* | 52.8 | 1.42 | 12.00 | 52.15 |

## Khoảng tin cậy 95% (bootstrap, 1,000 lần lấy lại)

Định dạng: trung bình [cận dưới, cận trên]. Faithfulness tính trên các mẫu được judge chấm.

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|
| `vit5_lora` | 28.01 [26.40, 29.59] | 13.73 [12.40, 15.10] | 20.70 [19.38, 22.11] | 55.34 [54.19, 56.45] |
| `vit5_lora_hier` | 16.30 [14.76, 18.07] | 7.91 [6.65, 9.28] | 13.48 [12.08, 15.01] | 46.77 [45.70, 47.88] |

## So sánh cặp với `vit5_lora` (paired bootstrap)

Hiệu = vit5_lora − hệ thống, tính trên cùng các mẫu. Khác biệt được coi là có ý nghĩa khi CI 95% của hiệu không chứa 0. Chưa hiệu chỉnh cho nhiều phép so sánh (multiple comparisons), nên các p sát ngưỡng cần đọc thận trọng.

| So với | Chỉ số | n | Hiệu | CI 95% | p | Kết luận |
|---|---|---|---|---|---|---|
| `vit5_lora_hier` | ROUGE-1 | 225 | +11.70 | [+9.89, +13.46] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier` | ROUGE-2 | 225 | +5.82 | [+4.37, +7.08] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier` | ROUGE-L | 225 | +7.22 | [+5.65, +8.70] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_lora_hier` | BERTScore-F | 225 | +8.56 | [+7.25, +9.89] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |

## Giới hạn của phương pháp đánh giá

- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp 10, input tách từ bằng underthesea (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ 256 subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa 225 mẫu/hệ thống (ngẫu nhiên, seed 42) bằng một model (claude-sonnet-4-6); độ tin cậy của judge được đối chiếu với người chấm trong `human_eval/agreement.md` (nếu đã chạy); judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn 30000 ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ 30000 ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm wikilingua_vi 125 mẫu, vietnews 100 mẫu; nhóm "long" (document > 1024 token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. Khoảng tin cậy và so sánh cặp ở trên dùng bootstrap trên toàn bộ tập, chưa tách theo dataset hay độ dài; các bảng theo tập con chỉ là trung bình điểm.

