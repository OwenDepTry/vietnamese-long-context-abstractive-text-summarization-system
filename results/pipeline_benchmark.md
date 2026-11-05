# Benchmark đường xử lý của API (3 mức độ dài như UI)

Sinh tự động lúc 2026-11-05 14:38 UTC bởi `scripts/benchmark_pipeline.py`. 40 văn bản test (20 ngắn, 20 dài > 1024 token), backend pytorch bf16 cuda. Config: strategy `auto`, hierarchical_above_tokens 1024, reduce `concat`, chunk_length `medium`, chunk_batch_size 4, chunk_cache_size 16; beam search như config (không phải streaming).

## Tổng thời gian chờ cho 3 mức độ dài (short + medium + long, như UI)

| Biến thể | Chiến lược | Số văn bản | p50 (ms) | p95 (ms) | Lần gọi model TB |
|---|---|---|---|---|---|
| cũ (chunk từng cái, không dùng lại) | direct | 20 | 2,721 | 3,363 | 3.0 |
| cũ (chunk từng cái, không dùng lại) | hierarchical | 20 | 3,906 | 6,248 | 6.3 |
| hiện tại (config) | direct | 20 | 2,806 | 3,545 | 3.0 |
| hiện tại (config) | hierarchical | 20 | 1,132 | 1,569 | 1.0 |

## Từng request

| Biến thể | Chiến lược | Số request | p50 (ms) | p95 (ms) | Lần gọi model TB | Dùng lại chunk (%) |
|---|---|---|---|---|---|---|
| cũ (chunk từng cái, không dùng lại) | direct | 60 | 860 | 1,426 | 1.0 | 0 |
| cũ (chunk từng cái, không dùng lại) | hierarchical | 60 | 1,292 | 2,095 | 2.1 | 0 |
| hiện tại (config) | direct | 60 | 918 | 1,488 | 1.0 | 0 |
| hiện tại (config) | hierarchical | 60 | 36 | 1,286 | 0.3 | 67 |

## Văn bản dài theo số chunk (tổng 3 mức độ dài)

| Biến thể | Số chunk | Số văn bản | p50 (ms) | Lần gọi model TB |
|---|---|---|---|---|
| cũ (chunk từng cái, không dùng lại) | 2 | 18 | 3,769 | 6.0 |
| cũ (chunk từng cái, không dùng lại) | 3 | 2 | 7,001 | 9.0 |
| hiện tại (config) | 2 | 18 | 1,108 | 1.0 |
| hiện tại (config) | 3 | 2 | 1,631 | 1.0 |

Bản tóm tắt giống hệt nhau giữa `cũ (chunk từng cái, không dùng lại)` và `hiện tại (config)`: **92.5%** số request. Khác biệt (nếu có) đến từ padding khi sinh chunk theo batch, không phải lỗi.

## Bộ nhớ

- RAM đỉnh của process (MB): 2,179
- VRAM tăng thêm, đỉnh (MB, nvidia-smi): 1,209

## Giới hạn

- Đo trong một process, gọi `Summarizer` trực tiếp: không gồm thời gian HTTP và hàng đợi của API.
- Request đồng thời chưa được đo: API mặc định `max_concurrency: 1`, nên request thứ hai chờ request đầu.
- Lần gọi đầu cho một văn bản luôn phải sinh chunk; lợi ích dùng lại chỉ có khi cùng văn bản được gọi nhiều mức độ dài (như UI).
