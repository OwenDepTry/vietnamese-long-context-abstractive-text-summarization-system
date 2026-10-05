# Phase 3 — Báo cáo đánh giá

Sinh tự động lúc 2026-10-05 18:00 UTC bởi `scripts/evaluate.py`. Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.

## Thiết lập

- Tập đánh giá: split `test`, vietnews: 1,000 mẫu, wikilingua_vi: 910 mẫu; nhóm long (document > 1024 token): 225 mẫu.
- Generation (model seq2seq): num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True.
- Hệ thống: `lead3` = 3 câu đầu của `document`; `vit5_lora` = ViT5-base + LoRA r=16 (phase 2), C:\vnsum-runs\vit5-vnsum-merged (input `input_text`); `vit5_public` = VietAI/vit5-base-vietnews-summarization (input `input_text`); `llm_zeroshot` = openai_compatible/gpt-6-luna zero-shot (input `document`).
- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).
- BERTScore: vinai/phobert-base, num_layers=10, tách từ underthesea, ≤ 256 subword, **raw (không rescale)**, thang 0–100.
- Faithfulness: LLM-as-Judge `claude-sonnet-4-6`, rubric 1–5, temperature 0, chấm trên cùng một mẫu ngẫu nhiên (seed 42): `lead3` 200 mẫu, `vit5_lora` 200 mẫu, `vit5_public` 200 mẫu, `llm_zeroshot` 200 mẫu.
- BERTScore: 33/15,280 chuỗi (prediction + reference) bị cắt ở 256 subword PhoBERT.
- Faithfulness: 11 kết quả judge có JSON không hợp lệ ({'vit5_lora': 6, 'vit5_public': 5}; chủ yếu do dấu " không escape trong unsupported_claims). Điểm được lấy từ trường `score` đứng đầu output, không gọi lại API; danh sách unsupported_claims của các mục này giữ nguyên dạng thô.

## Kết quả chính (toàn bộ tập đánh giá)

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) | Faithfulness (1–5) | n chấm |
|---|---|---|---|---|---|---|---|
| `lead3` | 1,910 | 24.91 | 11.41 | 17.17 | 51.15 | 5.00 | 200 |
| `vit5_lora` | 1,910 | 33.46 | 18.42 | 26.98 | 58.68 | 4.01 | 200 |
| `vit5_public` | 1,910 | 29.88 | 16.24 | 23.65 | 53.74 | 2.79 | 200 |
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
| `vit5_lora` | 11 | 27 | 21 | 31 | 110 | 0 | 4.01 |
| `vit5_public` | 39 | 59 | 42 | 26 | 34 | 0 | 2.79 |
| `llm_zeroshot` | 0 | 0 | 0 | 15 | 185 | 0 | 4.92 |

## Khoảng tin cậy 95% (bootstrap, 1,000 lần lấy lại)

Định dạng: trung bình [cận dưới, cận trên]. Faithfulness tính trên các mẫu được judge chấm.

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F | Faithfulness |
|---|---|---|---|---|---|
| `lead3` | 24.91 [24.49, 25.35] | 11.41 [11.02, 11.82] | 17.17 [16.80, 17.54] | 51.15 [50.85, 51.47] | 5.00 [5.00, 5.00] |
| `vit5_lora` | 33.46 [32.76, 34.12] | 18.42 [17.78, 18.98] | 26.98 [26.30, 27.57] | 58.68 [58.22, 59.09] | 4.01 [3.82, 4.19] |
| `vit5_public` | 29.88 [29.01, 30.81] | 16.24 [15.33, 17.16] | 23.65 [22.81, 24.57] | 53.74 [53.13, 54.37] | 2.79 [2.60, 2.96] |
| `llm_zeroshot` | 30.69 [30.23, 31.11] | 13.28 [12.91, 13.61] | 21.37 [21.00, 21.70] | 53.07 [52.79, 53.34] | 4.92 [4.88, 4.96] |

## So sánh cặp với `vit5_lora` (paired bootstrap)

Hiệu = vit5_lora − hệ thống, tính trên cùng các mẫu. Khác biệt được coi là có ý nghĩa khi CI 95% của hiệu không chứa 0. Chưa hiệu chỉnh cho nhiều phép so sánh (multiple comparisons), nên các p sát ngưỡng cần đọc thận trọng.

| So với | Chỉ số | n | Hiệu | CI 95% | p | Kết luận |
|---|---|---|---|---|---|---|
| `lead3` | ROUGE-1 | 1,910 | +8.56 | [+7.87, +9.26] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `lead3` | ROUGE-2 | 1,910 | +7.01 | [+6.42, +7.57] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `lead3` | ROUGE-L | 1,910 | +9.81 | [+9.20, +10.45] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `lead3` | BERTScore-F | 1,910 | +7.53 | [+7.09, +7.99] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `lead3` | Faithfulness | 200 | -0.99 | [-1.16, -0.81] | < 0.001 | có ý nghĩa: lead3 cao hơn |
| `vit5_public` | ROUGE-1 | 1,910 | +3.58 | [+2.62, +4.56] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_public` | ROUGE-2 | 1,910 | +2.18 | [+1.28, +3.06] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_public` | ROUGE-L | 1,910 | +3.33 | [+2.39, +4.24] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_public` | BERTScore-F | 1,910 | +4.94 | [+4.22, +5.62] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `vit5_public` | Faithfulness | 200 | +1.23 | [+0.99, +1.45] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `llm_zeroshot` | ROUGE-1 | 1,910 | +2.77 | [+2.13, +3.42] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `llm_zeroshot` | ROUGE-2 | 1,910 | +5.14 | [+4.61, +5.73] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `llm_zeroshot` | ROUGE-L | 1,910 | +5.61 | [+5.06, +6.19] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `llm_zeroshot` | BERTScore-F | 1,910 | +5.60 | [+5.19, +6.05] | < 0.001 | có ý nghĩa: vit5_lora cao hơn |
| `llm_zeroshot` | Faithfulness | 200 | -0.92 | [-1.09, -0.73] | < 0.001 | có ý nghĩa: llm_zeroshot cao hơn |

## Giới hạn của phương pháp đánh giá

- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp 10, input tách từ bằng underthesea (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ 256 subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa 200 mẫu/hệ thống (ngẫu nhiên, seed 42) bằng một model (claude-sonnet-4-6); độ tin cậy của judge được đối chiếu với người chấm trong `human_eval/agreement.md` (nếu đã chạy); judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn 30000 ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ 30000 ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm vietnews 1,000 mẫu, wikilingua_vi 910 mẫu; nhóm "long" (document > 1024 token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. Khoảng tin cậy và so sánh cặp ở trên dùng bootstrap trên toàn bộ tập, chưa tách theo dataset hay độ dài; các bảng theo tập con chỉ là trung bình điểm.

