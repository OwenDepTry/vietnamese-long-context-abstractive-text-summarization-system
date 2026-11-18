# vnsum — Tóm tắt trừu tượng văn bản tiếng Việt dài

[![tests](https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system/actions/workflows/tests.yml/badge.svg)](https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system/actions/workflows/tests.yml)

ViT5-base (`VietAI/vit5-base`) fine-tune bằng LoRA trên VietNews + WikiLingua (tiếng Việt), xử lý văn bản dài hơn ngữ cảnh 1024 token, đánh giá bằng ROUGE giữ dấu / BERTScore PhoBERT / LLM-as-Judge, tối ưu inference bằng ONNX Runtime và đóng gói thành API + UI chạy bằng Docker.

| Phase | Nội dung | Kết quả |
|---|---|---|
| 1 | Dữ liệu: tải, chuẩn hóa, lọc, xử lý văn bản dài | `data/processed/*.parquet` |
| 2 | Training: ViT5 + LoRA, merge | model đã merge (ngoài repo) |
| 3 | Evaluation: 4 hệ thống trên 1,910 mẫu test | [`results/main/report.md`](results/main/report.md), [`results/main/qualitative.md`](results/main/qualitative.md) |
| 4 | Inference: ONNX (KV cache) + INT8, benchmark CPU/GPU | [`results/inference/benchmark.md`](results/inference/benchmark.md) |
| 5 | API (FastAPI) + UI (Streamlit) + Docker | `src/vnsum/api/`, `ui/`, `docker-compose.yml` |
| 6 | Model card + đưa model lên Hugging Face Hub, demo trên Spaces, CI | `scripts/push_to_hub.py`, `demo/`, `.github/workflows/` |

## Kiến trúc

```
                 ┌──────────── offline ─────────────┐
 HF datasets ──► │ phase 1: chuẩn hóa, lọc, BM25    │──► data/processed/*.parquet
                 │ phase 2: ViT5 + LoRA ─► merge    │──► <MODEL_DIR>/vit5-vnsum-merged        (PyTorch)
                 │ phase 4: optimum export + INT8   │──► <MODEL_DIR>/vit5-vnsum-merged-onnx/  (fp32, int8)
                 │ phase 3/4: evaluate / benchmark  │──► results/main, results/inference
                 └──────────────────────────────────┘
                                   │ volume (read-only)
 trình duyệt ──► ui (Streamlit :8501) ──HTTP──► api (FastAPI :8000) ──► Summarizer
                                                   │                    ├─ tiền xử lý phase 1 (chuẩn hóa, extractive / hierarchical)
                                                   │                    └─ backend: ONNX Runtime (mặc định INT8, CPU) | PyTorch
                                                   └─ /summarize, /summarize/stream (SSE), /health
```

- `src/vnsum/inference/predictor.py` — `Summarizer`: một interface cho cả hai backend; dùng chung cho benchmark (phase 4) và API (phase 5).
- Model không nằm trong repo hay Docker image: đường dẫn đọc từ config / biến môi trường, thư mục model được mount vào container.
- Hai môi trường Python: `.venv` (phase 1–3, transformers 5) và `.venv-onnx` (phase 4–5, transformers 4.57.6 vì `optimum-onnx` 0.1.0 yêu cầu transformers < 4.58).

## Kết quả chính

**Phase 3** — 1,910 mẫu test (VietNews 1,000 + WikiLingua-vi 910), faithfulness chấm bởi `claude-sonnet-4-6` trên 200 mẫu/hệ thống. Chi tiết, khoảng tin cậy và so sánh cặp: [`results/main/report.md`](results/main/report.md).

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) | Faithfulness (1–5) |
|---|---|---|---|---|---|
| Lead-3 | 24.91 | 11.41 | 17.17 | 51.15 | 5.00 |
| **ViT5 + LoRA (dự án)** | **33.46** | **18.42** | **26.98** | **58.68** | 4.01 |
| ViT5 công khai (VietAI) | 29.88 | 16.24 | 23.65 | 53.74 | 2.79 |
| LLM zero-shot (gpt-6-luna) | 30.69 | 13.28 | 21.37 | 53.07 | 4.92 |

**Phase 4** — 100 tài liệu test, beam 4, max 128 token, batch 1, i7-13620H + RTX 4050 Laptop. Chi tiết (RAM/VRAM, theo độ dài input, chất lượng): [`results/inference/benchmark.md`](results/inference/benchmark.md).

| Thiết bị | Bản | p50 (ms) | p95 (ms) | Trên đĩa (MB) | ROUGE-L (Δ so với PyTorch) |
|---|---|---|---|---|---|
| CPU | PyTorch FP32 | 5,664 | 15,391 | 862 | 28.66 |
| CPU | ONNX INT8 | 4,875 | 10,405 | 418 | 29.07 (+0.41) |
| GPU | PyTorch bf16 | 2,284 | 4,335 | 862 | 28.45 |
| GPU | ONNX FP32 | 1,309 | 2,249 | 1,664 | 28.53 (+0.08) |

Mục tiêu latency < 500 ms **không đạt** ở cấu hình trên (kể cả theo nhóm độ dài input). Benchmark này dùng `strategy: extractive` (so backend trên cùng input). Đường xử lý thật của API (3 mức độ dài như UI, văn bản dài đi hierarchical concat) đo bằng `scripts/benchmark_pipeline.py` (xem [Phase 4](docs/phase4-inference.md)). INT8 giảm dung lượng trên đĩa (862 → 418 MB) nhưng gần như không giảm RAM đỉnh khi chạy.

**Các thí nghiệm mở rộng** (đều so với ViT5 + LoRA ở trên, cùng tập test):

| Thí nghiệm | Kết quả | Chi tiết |
|---|---|---|
| Văn bản dài: ghép bản tóm tắt chunk thay vì tóm tắt lại (`reduce: concat`) | 225 văn bản dài, ROUGE-L: concat **22.29**, extractive 20.70, extractive ép dài 20.55, tóm tắt lại 13.35 (chunk 128) / 13.48 (chunk 48). Cắt về độ dài của extractive: chưa đủ bằng chứng về khác biệt; so với extractive ép dài: concat hơn có ý nghĩa (xem [Giới hạn](docs/limitations.md)) | [`results/long_v2/report.md`](results/long_v2/report.md), [`length_control.md`](results/long_v2/length_control.md) |
| Tune `length_penalty` / `min_new_tokens` trên validation | +0.23 ROUGE-L, nằm trong nhiễu → giữ cấu hình cũ | [`results/inference/tune_lengths.md`](results/inference/tune_lengths.md) |
| Distillation: train tiếp trên 4,999 bản tóm tắt do LLM viết | **Kém hơn**: ROUGE-L 24.33 so với 26.98 (−2.65, CI 95% [2.08, 3.21]), kém ở mọi tập con; dài hơn (35.2 so với 26.8 âm tiết) nhưng lệch văn phong reference. Chưa chấm faithfulness | [`results/distill/report.md`](results/distill/report.md) |
| Đối chiếu LLM judge với người chấm (50 mục, chấm mù) | Kappa trọng số 0.70 (CI 95% [0.49, 0.85]), trùng khớp 66%, lệch ≤ 1 điểm 84% | [`results/main/human_eval/agreement.md`](results/main/human_eval/agreement.md) |
| Văn bản dài: chấm người theo cặp, concat so với extractive (20 văn bản, chấm mù, độc lập với judge) | **Đầy đủ** 2.70 so với 1.90 (+0.80, CI 95% [+0.50, +1.05], p < 0.001); **trung thực** 3.65 so với 4.25 (−0.60, CI [−1.25, −0.05], p 0.04): concat đầy đủ hơn nhưng sai sự thật nhiều hơn | [`results/long_v2/human_eval/agreement.md`](results/long_v2/human_eval/agreement.md) |
| Văn bản dài: đường xử lý thật của API (`vit5_lora_api`) | ROUGE-L 22.23, ngang concat của bên đánh giá (−0.06, CI [−0.41, +0.24]), hơn extractive +1.53 (CI [+0.40, +2.70]) | [`results/long_api/report.md`](results/long_api/report.md) |

Hai kết quả âm tính (tune độ dài, distillation) cho thấy làm bản tóm tắt dài hơn hoặc trừu tượng hơn **không** tự động cải thiện ROUGE / BERTScore so với reference (các chỉ số này không đo tính đúng sự thật).

## Chạy nhanh bằng Docker (phase 5)

Cần: Docker Desktop, model đã merge (phase 2) và bản ONNX (phase 4) trong cùng một thư mục, ví dụ `C:/vnsum-runs/vit5-vnsum-merged` và `C:/vnsum-runs/vit5-vnsum-merged-onnx/int8`.

```powershell
copy .env.example .env          # sửa MODEL_DIR nếu model nằm chỗ khác
docker compose build
docker compose up               # API: http://localhost:8000/docs   UI: http://localhost:8501
curl.exe -X POST localhost:8000/summarize -H "Content-Type: application/json" --data-binary "@examples/request.json"
```

## Tài liệu chi tiết

- [Phase 1 — Dữ liệu](docs/phase1-data.md): tải, chuẩn hóa, lọc, xử lý văn bản dài; cách chạy và thiết kế
- [Phase 2 — Training](docs/phase2-training.md): train ViT5 + LoRA, merge, kết quả train
- [Phase 3 — Evaluation](docs/phase3-evaluation.md): đánh giá 4 hệ thống, judge, chấm người, khoảng tin cậy
- [Phase 4 — Inference optimization](docs/phase4-inference.md): ONNX, INT8, benchmark, benchmark đường xử lý của API
- [Phase 5 — API, UI, Docker](docs/phase5-deployment.md): API FastAPI, UI Streamlit, Docker, test
- [Model trên Hugging Face Hub và demo](docs/phase6-hub-demo.md): model card, đưa model lên Hub, demo trên Spaces
- [Môi trường đã kiểm chứng](docs/environment.md): máy, phiên bản Python và thư viện đã chạy thật
- [Giới hạn còn lại](docs/limitations.md): các giới hạn đã biết và những gì đã sửa

## Cấu trúc

```
configs/            data.yaml (phase 1) · train*.yaml (2) · eval.yaml (3) · inference.yaml (4–5, gồm mục api)
scripts/            prepare_data · merge_lora · evaluate · human_eval · tune_lengths · length_control · distill_data · export_onnx
                    benchmark · benchmark_pipeline · profile_onnx · check_gpu_onnx · summarize · push_to_hub
src/vnsum/
  data/             phase 1: load, preprocess (NFC, dấu thanh, tách câu, lọc, dedup), long_context (BM25/LexRank, chunk), tokenization
  models/           phase 2: train (Seq2SeqTrainer + LoRA), lora, callbacks, rouge (tokenizer giữ dấu)
  eval/             phase 3: systems, metrics, bertscore, judge, llm_client, stats (bootstrap), report, human_eval
  inference/        phase 4: predictor (Summarizer), reduce (concat), export (ONNX/INT8/FP16), tokenizer, bench, pipeline_bench, hardware
  model_card.py     model card cho Hugging Face Hub, sinh từ results/main/metrics.csv
  api/              phase 5: app (FastAPI), streaming (SSE), settings
ui/app.py           phase 5: Streamlit, chỉ gọi API qua HTTP
docker/             api.Dockerfile (multi-stage, CPU, non-root) · ui.Dockerfile
demo/space/         demo trên Hugging Face Spaces (Docker: API + UI, model tải từ Hub)
.github/workflows/  CI: test offline (Python 3.11, 3.12) + ruff
docs/               chi tiết từng phase, môi trường, giới hạn (mục "Tài liệu chi tiết" ở trên)
notebooks/          train_colab (train trên Colab) · 01_eda (khám phá dữ liệu) · 02_error_analysis (phân tích lỗi)
tests/              pytest offline (tokenizer / backend giả); test model thật tự skip khi thiếu thư viện hoặc mạng
results/            kết quả theo thí nghiệm, mục lục ở results/README.md
  main/             phase 3: đánh giá chính 1,910 mẫu (report.md, qualitative.md, human_eval/, factcheck.md)
  inference/        phase 4: benchmark ONNX, pipeline_benchmark, tune_lengths, lengths
  long*/ distill/   văn bản dài (long, long_v2, long_api, long_fc) và distillation
requirements*.txt   requirements.txt (phase 1–3) · -inference (4) · -api (5) · -ui (5) · -ci (CI, không có torch) · -notebooks
```

## Giấy phép dữ liệu

WikiLingua: CC BY 3.0 (nội dung wikiHow). VietNews: xem bài báo VNDS (Nguyen et al., NICS 2019) và repo gốc `ThanhChinhBK/vietnews`.
