# Phase 3 — Báo cáo đánh giá

Sinh tự động lúc 2026-11-06 14:16 UTC bởi `scripts/evaluate.py`. Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.

## Thiết lập

- Tập đánh giá: split `test`, wikilingua_vi: 125 mẫu, vietnews: 100 mẫu; nhóm long (document > 1024 token): 225 mẫu.
- Generation (model seq2seq): num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True.
- Hệ thống: `vit5_lora_hier_concat` = ViT5 + LoRA, văn bản dài: tóm tắt từng chunk, ghép + bỏ câu trùng (không sinh lại), C:/vnsum-runs/vit5-vnsum-merged (input `input_text`); `vit5_lora_hier_concat_fc` = ViT5 + LoRA, văn bản dài: concat + bỏ câu có số/tên không căn cứ, C:/vnsum-runs/vit5-vnsum-merged (input `input_text`).
- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).
- BERTScore: vinai/phobert-base, num_layers=10, tách từ underthesea, ≤ 256 subword, **raw (không rescale)**, thang 0–100.
- BERTScore: 2/900 chuỗi (prediction + reference) bị cắt ở 256 subword PhoBERT.

**Không có trong kết quả:** `vit5_lora_hier_concat` (judge) (cần --allow_api); `vit5_lora_hier_concat_fc` (judge) (cần --allow_api).

## Kết quả chính (toàn bộ tập đánh giá)

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) |
|---|---|---|---|---|---|
| `vit5_lora_hier_concat` | 225 | 30.90 | 15.04 | 22.29 | 57.08 |
| `vit5_lora_hier_concat_fc` | 225 | 30.83 | 15.04 | 22.27 | 57.03 |

## Theo dataset

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora_hier_concat` | vietnews | 100 | 27.84 | 13.70 | 19.49 | 51.65 |
| `vit5_lora_hier_concat` | wikilingua_vi | 125 | 33.35 | 16.12 | 24.53 | 61.43 |
| `vit5_lora_hier_concat_fc` | vietnews | 100 | 27.70 | 13.70 | 19.45 | 51.51 |
| `vit5_lora_hier_concat_fc` | wikilingua_vi | 125 | 33.34 | 16.11 | 24.53 | 61.44 |

## Theo độ dài document (long vs short)

| Hệ thống | Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|---|---|
| `vit5_lora_hier_concat` | long | 225 | 30.90 | 15.04 | 22.29 | 57.08 |
| `vit5_lora_hier_concat_fc` | long | 225 | 30.83 | 15.04 | 22.27 | 57.03 |

## Chỉ số phụ (toàn bộ tập đánh giá)

Lặp = tỷ lệ n-gram trong summary là bản lặp của n-gram đứng trước. Novel = tỷ lệ n-gram của summary không có trong document gốc. Số/tên không căn cứ = % bản tóm tắt có số, ngày hoặc tên riêng không tìm thấy trong văn bản gốc (`vnsum.factcheck`; chỉ bắt lỗi bề mặt, không bắt lỗi quan hệ như gán nhầm lời nói).

|  | Độ dài TB (âm tiết) | repetition 3gram (%) | novel 1gram (%) | novel 2gram (%) | số/tên không căn cứ (%) |
|---|---|---|---|---|---|
| `vit5_lora_hier_concat` | 54.0 | 1.86 | 1.59 | 18.81 | 6.22 |
| `vit5_lora_hier_concat_fc` | 52.1 | 1.68 | 1.33 | 18.07 | 0.00 |
| *reference (tham chiếu)* | 52.8 | 1.42 | 12.00 | 52.15 | 18.67 |

## Khoảng tin cậy 95% (bootstrap, 1,000 lần lấy lại)

Định dạng: trung bình [cận dưới, cận trên]. Faithfulness tính trên các mẫu được judge chấm.

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F |
|---|---|---|---|---|
| `vit5_lora_hier_concat` | 30.90 [29.50, 32.31] | 15.04 [13.81, 16.34] | 22.29 [20.99, 23.60] | 57.08 [56.14, 57.96] |
| `vit5_lora_hier_concat_fc` | 30.83 [29.36, 32.27] | 15.04 [13.71, 16.34] | 22.27 [20.87, 23.58] | 57.03 [56.10, 57.96] |

## Giới hạn của phương pháp đánh giá

- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp 10, input tách từ bằng underthesea (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ 256 subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa 200 mẫu/hệ thống (ngẫu nhiên, seed 42) bằng một model (claude-sonnet-4-6); độ tin cậy của judge được đối chiếu với người chấm trong `human_eval/agreement.md` (nếu đã chạy); judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn 30000 ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ 30000 ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm wikilingua_vi 125 mẫu, vietnews 100 mẫu; nhóm "long" (document > 1024 token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. Khoảng tin cậy và so sánh cặp ở trên lấy lại theo bài gốc (các mục WikiLingua cùng bài đi cùng nhau), phân tầng theo dataset; các bảng theo tập con chỉ là trung bình điểm.

