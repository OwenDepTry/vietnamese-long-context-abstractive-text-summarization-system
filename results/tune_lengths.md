# Bước 2 — tune độ dài trên tập validation

Sinh tự động lúc 2026-10-30 11:56 UTC bởi `scripts/tune_lengths.py`. 300 mẫu **validation** (vietnews: 150, wikilingua_vi: 150), input `input_text` như lúc train, backend pytorch bf16 cuda. Mức `medium`: num_beams=4, no_repeat_ngram_size=3, max_new_tokens=128. Ứng viên bị loại nếu lặp 3-gram cao hơn baseline quá 1 điểm.

Độ dài TB của reference: **39.4 âm tiết**.

| Ứng viên | length_penalty | min_new_tokens | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-L vietnews | ROUGE-L wikilingua_vi | Âm tiết TB | Lặp 3-gram (%) |
|---|---|---|---|---|---|---|---|---|---|
| `lp1_min0` (hiện tại) | 1 | 0 | 32.74 | 17.24 | 26.56 | 27.43 | 25.68 | 26.6 | 0.05 |
| `lp1.5_min0` | 1.5 | 0 | 33.35 | 17.53 | 26.75 | 27.53 | 25.97 | 28.5 | 0.07 |
| `lp2_min0` | 2 | 0 | 33.35 | 17.40 | 26.49 | 27.35 | 25.63 | 29.3 | 0.07 |
| `lp1_min16` | 1 | 16 | 33.12 | 17.48 | 26.72 | 27.43 | 26.01 | 27.0 | 0.05 |
| `lp1.5_min16` **(chọn)** | 1.5 | 16 | 33.57 | 17.67 | 26.79 | 27.53 | 26.05 | 28.9 | 0.07 |
| `lp2_min16` | 2 | 16 | 33.65 | 17.59 | 26.69 | 27.35 | 26.03 | 29.8 | 0.07 |
| `lp1_min24` | 1 | 24 | 33.58 | 17.57 | 26.78 | 27.33 | 26.24 | 28.6 | 0.15 |
| `lp1.5_min24` | 1.5 | 24 | 33.80 | 17.62 | 26.77 | 27.40 | 26.13 | 30.5 | 0.15 |
| `lp2_min24` | 2 | 24 | 33.92 | 17.59 | 26.70 | 27.17 | 26.22 | 31.3 | 0.18 |
| `lp1_min32` | 1 | 32 | 33.49 | 17.26 | 26.34 | 26.81 | 25.87 | 32.5 | 0.14 |
| `lp1.5_min32` | 1.5 | 32 | 33.60 | 17.39 | 26.23 | 26.71 | 25.76 | 34.1 | 0.12 |
| `lp2_min32` | 2 | 32 | 33.73 | 17.41 | 26.22 | 26.71 | 25.74 | 34.9 | 0.13 |

## Kết luận

- Chọn `lp1.5_min16`: ROUGE-L 26.79 so với 26.56 của cấu hình hiện tại (+0.23), độ dài 26.6 → 28.9 âm tiết.
- Đây là số trên VALIDATION. Sau khi cập nhật config, chạy lại đánh giá trên test để có số cuối cùng.

Bước tiếp theo: chép `min_new_tokens` / `length_penalty` dưới đây vào `configs/eval.yaml` (mục `systems.vit5_lora_tuned.generation`), đánh giá trên test; nếu vẫn tốt hơn thì cập nhật `configs/inference.yaml` (mục `lengths.medium`):

```yaml
medium:
  max_new_tokens: 128
  min_new_tokens: 16
  length_penalty: 1.5
```
