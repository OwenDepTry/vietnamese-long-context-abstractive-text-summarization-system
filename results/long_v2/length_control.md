# Đối chứng độ dài

Sinh tự động lúc 2026-11-05 15:56 UTC bởi `scripts/length_control.py`, 225 mẫu. Dự đoán của `vit5_lora_hier_concat` được cắt về đúng độ dài (âm tiết) của dự đoán đối chứng trên cùng mẫu (chỉ cắt khi dài hơn), chấm lại ROUGE, rồi so cặp: hiệu = `vit5_lora_hier_concat` − đối chứng, CI 95% paired bootstrap (1,000 lần).

## So với `vit5_lora` (độ dài TB 30.8 âm tiết)

| Biến thể | Âm tiết TB | ROUGE-L | Hiệu ROUGE-L | Hiệu ROUGE-2 |
|---|---|---|---|---|
| `vit5_lora` (đối chứng) | 30.8 | 20.70 | | |
| `vit5_lora_hier_concat` nguyên bản | 54.0 | 22.29 | +1.59 [+0.49, +2.72], p 0.006 | +1.31 [+0.28, +2.41], p 0.008 |
| `vit5_lora_hier_concat` cắt theo âm tiết | 30.6 | 21.03 | +0.34 [-0.66, +1.35], p 0.502 | -0.24 [-1.18, +0.68], p 0.636 |
| `vit5_lora_hier_concat` cắt theo câu | 31.1 | 21.10 | +0.40 [-0.65, +1.44], p 0.462 | -0.21 [-1.18, +0.73], p 0.658 |

## So với `vit5_lora_tuned` (độ dài TB 60.7 âm tiết)

| Biến thể | Âm tiết TB | ROUGE-L | Hiệu ROUGE-L | Hiệu ROUGE-2 |
|---|---|---|---|---|
| `vit5_lora_tuned` (đối chứng) | 60.7 | 20.55 | | |
| `vit5_lora_hier_concat` nguyên bản | 54.0 | 22.29 | +1.74 [+0.69, +2.86], p < 0.001 | +1.14 [+0.08, +2.23], p 0.032 |
| `vit5_lora_hier_concat` cắt theo âm tiết | 49.0 | 22.24 | +1.69 [+0.63, +2.73], p < 0.001 | +1.06 [+0.05, +2.16], p 0.046 |
| `vit5_lora_hier_concat` cắt theo câu | 49.2 | 22.29 | +1.74 [+0.71, +2.82], p < 0.001 | +1.02 [+0.01, +2.12], p 0.050 |

Đọc: nếu hiệu chỉ có ý nghĩa ở bản nguyên bản mà mất khi cắt về cùng độ dài, phần thắng chủ yếu đến từ độ dài. Cắt theo âm tiết có thể cắt giữa câu (hơi bất lợi cho hệ thống bị cắt); cắt theo câu giữ câu trọn vẹn nhưng độ dài chỉ gần đúng.
