# Kết quả

Mỗi thư mục là kết quả của một lần chạy hoặc một nhóm thí nghiệm. Thư mục do `scripts/evaluate.py` sinh ra có cùng bộ file: `report.md` (báo cáo), `qualitative.md` (ví dụ), `metrics.csv`, `per_sample.csv`, `significance.csv`, `predictions/` (cache dự đoán) và `run_info.json` (commit, config, phiên bản thư viện; chỉ có từ các lần chạy sau này).

| Thư mục | Nội dung | Đọc trước |
|---|---|---|
| [`main/`](main/) | Phase 3: đánh giá chính, 4 hệ thống trên 1,910 mẫu test; đối chiếu judge với người chấm (`human_eval/`, 50 mục); kiểm chứng factcheck (`factcheck.md`) | [`report.md`](main/report.md) |
| [`inference/`](inference/) | Phase 4: benchmark PyTorch / ONNX / INT8 (`benchmark.md`, số thô trong `benchmark/`), đo đường xử lý của UI (`pipeline_benchmark.*`), tune `length_penalty` / `min_new_tokens` (`tune_lengths.*`, `tune_lengths/`), so sánh ba mức độ dài (`lengths.md`) | [`benchmark.md`](inference/benchmark.md) |
| [`long/`](long/) | Văn bản dài, bản đầu: so sánh extractive, tóm tắt lại (hier) và ghép (concat) trên 225 văn bản | [`report.md`](long/report.md) |
| [`long_v2/`](long_v2/) | Văn bản dài, đối chứng độ dài (`length_control.md`) và chấm người theo cặp (`human_eval/`, 20 văn bản) | [`report.md`](long_v2/report.md) |
| [`long_api/`](long_api/) | Văn bản dài, chạy đúng đường xử lý của API (`vit5_lora_api`) | [`report.md`](long_api/report.md) |
| [`long_fc/`](long_fc/) | Văn bản dài, bỏ câu bị factcheck đánh dấu trong concat | [`report.md`](long_fc/report.md) |
| [`distill/`](distill/) | Distillation: train tiếp trên bản tóm tắt do LLM viết | [`report.md`](distill/report.md) |

Smoke test ghi vào thư mục con `smoke/` (vd. `main/smoke/`, `inference/smoke/`) và không được commit (xem `.gitignore`).
