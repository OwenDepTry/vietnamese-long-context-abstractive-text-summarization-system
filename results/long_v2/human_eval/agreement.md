# Kết quả người chấm

Đã chấm 40/40 mục, chấm mù (không biết tên hệ thống). Các mục này chưa có điểm judge nên chưa tính đồng thuận người–judge.

## Điểm người chấm theo hệ thống

| Hệ thống | n | Trung thực TB | Đầy đủ TB |
|---|---:|---:|---:|
| vit5_lora | 20 | 4.25 | 1.90 |
| vit5_lora_hier_concat | 20 | 3.65 | 2.70 |

## So sánh cặp trên cùng văn bản (paired bootstrap)

| So sánh | Chỉ số | n văn bản | Hiệu | CI 95% | p |
|---|---|---:|---:|---|---:|
| vit5_lora_hier_concat − vit5_lora | Trung thực | 20 | -0.60 | [-1.25, -0.05] | 0.042 |
| vit5_lora_hier_concat − vit5_lora | Đầy đủ | 20 | +0.80 | [+0.50, +1.05] | < 0.001 |

Chỉ tính trên văn bản mà cả hai hệ thống đều đã được chấm (export với `--paired`).
