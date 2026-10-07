# Phase 4 — Benchmark inference

Sinh tự động lúc 2026-10-07 17:48 UTC bởi `scripts/benchmark.py`. Mọi con số dưới đây là số đo thật của lần chạy này (file JSON từng lượt chạy nằm cạnh file này).

Lệnh: `python scripts/benchmark.py --config configs/inference.yaml --model_path C:\vnsum-runs\vit5-vnsum-merged --runs pytorch_cpu,onnx_fp32_cpu,onnx_int8_cpu,pytorch_cuda,onnx_fp32_cuda`

## Điều kiện đo

- **Tài liệu:** 100 tài liệu test cố định (seed 42) lấy từ tập đánh giá phase 3, giống nhau cho mọi lượt chạy; 5 lần warmup (không tính) trước khi đo.
- **Input:** chiến lược `extractive` (BM25 extractive filter ≤ 1024 token như lúc train / phase 3).
- **Generation:** batch 1, num_beams=4, max_new_tokens=128 (mức `medium`), no_repeat_ngram_size=3, early_stopping=True.
- **Latency:** thời gian end-to-end một tài liệu (chuẩn hóa + chọn câu + tokenize + generate + decode), đo bằng `time.perf_counter`; throughput = số tài liệu / tổng thời gian khi chạy tuần tự.
- **Peak RAM:** RAM đỉnh của process chạy lượt đó (mỗi lượt chạy trong một process riêng). **Peak VRAM:** mức tăng `memory.used` của GPU (nvidia-smi) so với trước khi nạp model.

## Phần cứng

- **CPU:** 13th Gen Intel(R) Core(TM) i7-13620H — 16 luồng logic; PyTorch dùng 10 luồng
- **RAM:** 15.7 GB
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB, driver 610.78); CUDA của torch: 12.8
- **Hệ điều hành:** Windows 11 (10.0.26200) (AMD64)
- **ONNX Runtime providers:** TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
- **Thư viện:** python 3.13.12, torch 2.9.1+cu128, transformers 4.57.6, optimum 2.1.0, optimum-onnx 0.1.0, onnxruntime-gpu 1.23.2, onnx 1.23.2

## Kết quả CPU

| Lượt chạy | Bản | dtype | p50 (ms) | p95 (ms) | Tài liệu/s | Token sinh/s | Peak RAM (MB) | Trên đĩa (MB) | ROUGE-L (Δ so với PyTorch) | Tăng tốc p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| `pytorch_cpu` | PyTorch (trước) | fp32 | 5,664 | 15,391 | 0.15 | 4.9 | 2,144 | 862 | 28.66 | 1.00× |
| `onnx_fp32_cpu` | ONNX FP32 | fp32 | 6,474 | 12,281 | 0.15 | 4.8 | 3,338 | 1,664 | 28.66 (+0.00) | 0.87× |
| `onnx_int8_cpu` | ONNX INT8 dynamic | int8 | 4,875 | 10,405 | 0.16 | 4.9 | 2,152 | 418 | 29.07 (+0.41) | 1.16× |

Số luồng CPU: `pytorch_cpu` 10, `onnx_fp32_cpu` mặc định ORT, `onnx_int8_cpu` mặc định ORT.

## Kết quả GPU

| Lượt chạy | Bản | dtype | p50 (ms) | p95 (ms) | Tài liệu/s | Token sinh/s | Peak VRAM (MB) | Trên đĩa (MB) | ROUGE-L (Δ so với PyTorch) | Tăng tốc p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| `pytorch_cuda` | PyTorch (trước) | bf16 | 2,284 | 4,335 | 0.41 | 13.7 | 997 | 862 | 28.45 | 1.00× |
| `onnx_fp32_cuda` | ONNX FP32 | fp32 | 1,309 | 2,249 | 0.73 | 24.2 | 3,738 | 1,664 | 28.53 (+0.08) | 1.75× |

PyTorch allocator: `pytorch_cuda`: torch.cuda.max_memory_allocated = 623 MB.

## Latency theo độ dài input (p50 / p95, ms)

Độ dài = số token ViT5 thực sự đưa vào model (sau extractive filter, gồm `</s>`).

| Lượt chạy | input ≤256 token | input 257–512 token | input >512 token |
|---|---|---|---|
| `pytorch_cpu` | 1,624 / 6,817 (n=16) | 5,982 / 12,713 (n=40) | 8,671 / 18,741 (n=44) |
| `onnx_fp32_cpu` | 3,712 / 5,797 (n=16) | 5,721 / 9,243 (n=40) | 8,231 / 14,099 (n=44) |
| `onnx_int8_cpu` | 2,815 / 6,209 (n=16) | 4,145 / 7,245 (n=40) | 6,766 / 12,140 (n=44) |
| `pytorch_cuda` | 1,905 / 2,832 (n=16) | 2,246 / 3,823 (n=40) | 2,553 / 4,368 (n=44) |
| `onnx_fp32_cuda` | 1,027 / 1,992 (n=16) | 1,324 / 2,005 (n=40) | 1,326 / 2,379 (n=44) |

## Mục tiêu latency < 500 ms

| Lượt chạy | p50 (ms) | p50 < mục tiêu | p95 (ms) | p95 < mục tiêu |
|---|---|---|---|---|
| `pytorch_cpu` | 5,664 | không đạt | 15,391 | không đạt |
| `onnx_fp32_cpu` | 6,474 | không đạt | 12,281 | không đạt |
| `onnx_int8_cpu` | 4,875 | không đạt | 10,405 | không đạt |
| `pytorch_cuda` | 2,284 | không đạt | 4,335 | không đạt |
| `onnx_fp32_cuda` | 1,309 | không đạt | 2,249 | không đạt |

**Không lượt chạy nào đạt p95 < 500 ms trên toàn bộ tài liệu đo** (beam 4, max_new_tokens 128, batch 1).

Theo nhóm độ dài input, p95 < mục tiêu ở: không nhóm nào.

## Suy giảm chất lượng (ROUGE-L, cùng tài liệu)

Ngưỡng cảnh báo: giảm quá 1.0 điểm ROUGE-L so với PyTorch cùng thiết bị.

| Bản | So với | ROUGE-L | ROUGE-L PyTorch | Hiệu | Output giống hệt | Kết luận |
|---|---|---|---|---|---|---|
| `onnx_fp32_cpu` | `pytorch_cpu` | 28.66 | 28.66 | +0.00 | 100/100 | trong ngưỡng |
| `onnx_int8_cpu` | `pytorch_cpu` | 29.07 | 28.66 | +0.41 | 13/100 | trong ngưỡng |
| `onnx_fp32_cuda` | `pytorch_cuda` | 28.53 | 28.45 | +0.08 | 87/100 | trong ngưỡng |

## Giới hạn

- Đo trên một máy, batch 1, chạy tuần tự; laptop có thể giảm xung khi nóng hoặc chạy pin, nên số đo dao động giữa các lần.
- Peak VRAM đọc từ nvidia-smi là bộ nhớ của cả GPU (trừ mức nền); chương trình khác dùng GPU trong lúc đo sẽ làm sai số này.
- ROUGE-L chỉ đo trùng lặp bề mặt với reference; output giống hệt PyTorch là chỉ báo trực tiếp hơn cho việc chuyển đổi có giữ nguyên hành vi hay không.
