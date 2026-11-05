# Đối chứng độ dài

Sinh tự động lúc 2026-11-05 14:13 UTC bởi `scripts/length_control.py`, 225 mẫu. Dự đoán của `vit5_lora_hier_concat` được cắt về đúng độ dài (âm tiết) của dự đoán đối chứng trên cùng mẫu (chỉ cắt khi dài hơn), chấm lại ROUGE, rồi so cặp: hiệu = `vit5_lora_hier_concat` − đối chứng, CI 95% paired bootstrap (1,000 lần).

## So với `vit5_lora` (độ dài TB 30.8 âm tiết)

| Biến thể | Âm tiết TB | ROUGE-L | Hiệu ROUGE-L | Hiệu ROUGE-2 |
|---|---|---|---|---|
| `vit5_lora` (đối chứng) | 30.8 | 20.70 | | |
| `vit5_lora_hier_concat` nguyên bản | 53.9 | 22.26 | +1.56 [+0.46, +2.69], p 0.006 | +1.32 [+0.28, +2.40], p 0.008 |
| `vit5_lora_hier_concat` cắt theo âm tiết | 30.6 | 21.03 | +0.33 [-0.67, +1.35], p 0.514 | -0.23 [-1.18, +0.68], p 0.636 |
| `vit5_lora_hier_concat` cắt theo câu | 31.1 | 21.09 | +0.39 [-0.66, +1.44], p 0.464 | -0.21 [-1.18, +0.73], p 0.658 |

## So với `vit5_lora_tuned` (độ dài TB 60.7 âm tiết)

| Biến thể | Âm tiết TB | ROUGE-L | Hiệu ROUGE-L | Hiệu ROUGE-2 |
|---|---|---|---|---|
| `vit5_lora_tuned` (đối chứng) | 60.7 | 20.55 | | |
| `vit5_lora_hier_concat` nguyên bản | 53.9 | 22.26 | +1.71 [+0.67, +2.81], p < 0.001 | +1.14 [+0.09, +2.24], p 0.032 |
| `vit5_lora_hier_concat` cắt theo âm tiết | 48.9 | 22.21 | +1.66 [+0.62, +2.71], p < 0.001 | +1.06 [+0.05, +2.17], p 0.044 |
| `vit5_lora_hier_concat` cắt theo câu | 49.1 | 22.26 | +1.71 [+0.68, +2.81], p < 0.001 | +1.02 [+0.01, +2.13], p 0.048 |

Đọc: nếu hiệu chỉ có ý nghĩa ở bản nguyên bản mà mất khi cắt về cùng độ dài, phần thắng chủ yếu đến từ độ dài. Cắt theo âm tiết có thể cắt giữa câu (hơi bất lợi cho hệ thống bị cắt); cắt theo câu giữ câu trọn vẹn nhưng độ dài chỉ gần đúng.
