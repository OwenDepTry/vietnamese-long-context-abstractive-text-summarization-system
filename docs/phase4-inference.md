# Phase 4 — Inference optimization

← [Về README](../README.md)

Chạy trong `.venv-onnx`:

```powershell
python -m venv .venv-onnx
.venv-onnx\Scripts\activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-inference.txt
python -m pytest tests/test_inference.py

python scripts/export_onnx.py --config configs/inference.yaml --model_path <model đã merge>
python scripts/benchmark.py --config configs/inference.yaml --model_path <model đã merge> --limit 5 --device cpu   # smoke
python scripts/benchmark.py --config configs/inference.yaml --model_path <model đã merge> --runs pytorch_cpu,onnx_fp32_cpu,onnx_int8_cpu,pytorch_cuda,onnx_fp32_cuda
```

- **Export:** `optimum` export encoder + decoder + decoder_with_past (KV cache); mỗi file được lượng tử hóa dynamic INT8 bằng `ORTQuantizer`. File ONNX nằm cạnh model gốc (`<model>-onnx/{fp32,int8}`), không nằm trong repo. ONNX FP32 cho output giống hệt PyTorch ở 100/100 tài liệu benchmark.
- **`Summarizer`** (`src/vnsum/inference/predictor.py`): một interface cho backend PyTorch và ONNX, ba mức độ dài `short/medium/long` (max_new_tokens 48/128/256; short thêm `length_penalty` 0.6, long thêm `min_new_tokens` 64 và `length_penalty` 2.0; medium giữ đúng cấu hình phase 3), văn bản dài xử lý theo `long_document.strategy` (đưa thẳng / extractive filter như lúc train / hierarchical: tóm tắt từng chunk rồi tóm tắt bản ghép).
- **Benchmark:** cùng 100 tài liệu test và generation config với phase 3, 5 lần warmup, mỗi lượt chạy một process riêng; đo p50/p95, throughput, RAM/VRAM đỉnh, kích thước trên đĩa và ROUGE-L so với PyTorch (cảnh báo khi giảm quá 1 điểm); ghi cấu hình phần cứng thật.
- **Spin-wait của ONNX Runtime** được tắt (`onnx.allow_spinning: false`): khi bật, thread pool của ORT giành CPU với PyTorch trong vòng beam search (đo bằng `scripts/profile_onnx.py`: 3,829 → 1,683 ms trên 2 tài liệu).
- **Đo thêm cấu hình:** `--num_beams 1` (greedy), `--length short`, `--tag <tên>` để ghi ra `results/inference/benchmark_<tên>.md` mà không ghi đè bản chính.
- **FP16 không dùng được:** chuyển FP32 → FP16 (2 lần thử, lần 2 giữ LayerNorm/residual/softmax/FFN đầu ra ở FP32) đều cho chuỗi vô nghĩa, trong khi ONNX FP32 trên cùng GPU cho kết quả đúng (`scripts/check_gpu_onnx.py`). Cờ `--fp16` vẫn còn nhưng không nằm trong benchmark.

## Benchmark đường xử lý của API

```powershell
python scripts/benchmark_pipeline.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
# nhanh: --n_short 5 --n_long 5 ; ONNX: --backend onnx
```

Mô phỏng đúng việc UI làm: 3 request (short, medium, long) cho cùng văn bản, config của API (văn bản > 1024 token đi
hierarchical concat). So hai biến thể trên cùng tài liệu: **cũ** (tóm tắt chunk từng cái, mỗi mức độ dài sinh lại) và
**hiện tại** (`long_document.chunk_batch_size`: sinh chunk theo batch; `chunk_cache_size`: nhớ bản tóm tắt chunk của
văn bản gần nhất, nên mức độ dài thứ 2 và 3 không sinh lại). Báo latency theo chiến lược, tổng thời gian chờ cho 3
mức, theo số chunk, và % bản tóm tắt giống hệt nhau giữa hai biến thể → `results/inference/pipeline_benchmark.md`.

Kết quả (40 văn bản test: 20 ngắn, 20 dài; PyTorch bf16, RTX 4050 Laptop; [`results/inference/pipeline_benchmark.md`](../results/inference/pipeline_benchmark.md)):

| Tổng thời gian chờ cho 3 mức độ dài | Cũ: p50 / p95 (ms) | Hiện tại: p50 / p95 (ms) | Lần gọi model (cũ → hiện tại) |
|---|---|---|---|
| Văn bản ngắn (direct) | 2,721 / 3,363 | 2,806 / 3,545 | 3.0 → 3.0 |
| Văn bản dài (hierarchical, 2–3 chunk) | 3,906 / 6,248 | **1,132 / 1,569** | 6.3 → 1.0 |

Văn bản dài nhanh hơn ~3.4 lần **trong phép đo này** (PyTorch GPU, gọi `Summarizer` trực tiếp, luồng 3 mức độ dài;
không gồm HTTP, hàng đợi, và chưa đo ONNX CPU): các chunk sinh chung một batch, mức medium và long dùng lại bản tóm
tắt chunk. **Output có đổi:** văn bản ngắn giống hệt 60/60 request, văn bản dài chỉ 51/60 (3/20 văn bản đổi, mỗi văn
bản đổi cả 3 mức). Dùng lại chunk không đổi nội dung, nên khác biệt đến từ việc sinh theo batch; cơ chế cụ thể
(padding, sai số bf16) chưa được kiểm chứng. Ảnh hưởng chất lượng đo trên đủ 225 văn bản dài
(`results/long_v2/`): sinh chunk theo batch (`vit5_lora_hier_concat`) và từng chunk một (`vit5_lora_hier_concat_b1`)
cho output giống hệt ở 74.2% văn bản, nhưng ROUGE-L chênh **+0.00** (CI 95% [−0.32, +0.36], p 0.96); ROUGE-1/2 và
BERTScore cũng không khác biệt; trên 58 văn bản có output khác, bản batch tốt hơn ở 26. Không phát hiện ảnh hưởng
chất lượng đo được bằng ROUGE / BERTScore (chưa đo faithfulness). Xu hướng giảm trên 20 văn bản của benchmark
(20.18 → 19.84, p 0.06) là mẫu nhỏ. Lưu ý: đối chứng này dùng batch của bên đánh giá (8, gom chunk nhiều văn bản), khác API (`chunk_batch_size` 4, trong từng văn bản). Hệ thống `vit5_lora_api` (type `pipeline`) chạy **đúng** `Summarizer` + `configs/inference.yaml` trên văn bản gốc, từng văn bản như request: ROUGE-L 22.23 so với 22.29 của concat bên đánh giá (−0.06, CI 95% [−0.41, +0.24], p 0.77; 174/225 output giống hệt), vẫn hơn extractive +1.53 (CI [+0.40, +2.70]) (`results/long_api/`). `--ablation` tách 4 cấu hình batch 1/4 × có/không dùng lại (chưa chạy). Một request đơn lẻ vẫn mất ~0.9 s (p50, beam 4), chưa
đạt mục tiêu < 500 ms. Chưa đo trên ONNX INT8 / CPU (cấu hình của Docker và demo).
