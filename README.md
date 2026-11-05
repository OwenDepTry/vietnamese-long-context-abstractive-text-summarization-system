# vnsum — Tóm tắt trừu tượng văn bản tiếng Việt dài

[![tests](https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system/actions/workflows/tests.yml/badge.svg)](https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system/actions/workflows/tests.yml)

ViT5-base (`VietAI/vit5-base`) fine-tune bằng LoRA trên VietNews + WikiLingua (tiếng Việt), xử lý văn bản dài hơn ngữ cảnh 1024 token, đánh giá bằng ROUGE giữ dấu / BERTScore PhoBERT / LLM-as-Judge, tối ưu inference bằng ONNX Runtime và đóng gói thành API + UI chạy bằng Docker.

| Phase | Nội dung | Kết quả |
|---|---|---|
| 1 | Dữ liệu: tải, chuẩn hóa, lọc, xử lý văn bản dài | `data/processed/*.parquet` |
| 2 | Training: ViT5 + LoRA, merge | model đã merge (ngoài repo) |
| 3 | Evaluation: 4 hệ thống trên 1,910 mẫu test | [`results/report.md`](results/report.md), [`results/qualitative.md`](results/qualitative.md) |
| 4 | Inference: ONNX (KV cache) + INT8, benchmark CPU/GPU | [`results/benchmark.md`](results/benchmark.md) |
| 5 | API (FastAPI) + UI (Streamlit) + Docker | `src/vnsum/api/`, `ui/`, `docker-compose.yml` |
| 6 | Model card + đưa model lên Hugging Face Hub, demo trên Spaces, CI | `scripts/push_to_hub.py`, `demo/`, `.github/workflows/` |

## Kiến trúc

```
                 ┌──────────── offline ─────────────┐
 HF datasets ──► │ phase 1: chuẩn hóa, lọc, BM25    │──► data/processed/*.parquet
                 │ phase 2: ViT5 + LoRA ─► merge    │──► <MODEL_DIR>/vit5-vnsum-merged        (PyTorch)
                 │ phase 4: optimum export + INT8   │──► <MODEL_DIR>/vit5-vnsum-merged-onnx/  (fp32, int8)
                 │ phase 3/4: evaluate / benchmark  │──► results/*.md
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

**Phase 3** — 1,910 mẫu test (VietNews 1,000 + WikiLingua-vi 910), faithfulness chấm bởi `claude-sonnet-4-6` trên 200 mẫu/hệ thống. Chi tiết, khoảng tin cậy và so sánh cặp: [`results/report.md`](results/report.md).

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) | Faithfulness (1–5) |
|---|---|---|---|---|---|
| Lead-3 | 24.91 | 11.41 | 17.17 | 51.15 | 5.00 |
| **ViT5 + LoRA (dự án)** | **33.46** | **18.42** | **26.98** | **58.68** | 4.01 |
| ViT5 công khai (VietAI) | 29.88 | 16.24 | 23.65 | 53.74 | 2.79 |
| LLM zero-shot (gpt-6-luna) | 30.69 | 13.28 | 21.37 | 53.07 | 4.92 |

**Phase 4** — 100 tài liệu test, beam 4, max 128 token, batch 1, i7-13620H + RTX 4050 Laptop. Chi tiết (RAM/VRAM, theo độ dài input, chất lượng): [`results/benchmark.md`](results/benchmark.md).

| Thiết bị | Bản | p50 (ms) | p95 (ms) | Trên đĩa (MB) | ROUGE-L (Δ so với PyTorch) |
|---|---|---|---|---|---|
| CPU | PyTorch FP32 | 5,664 | 15,391 | 862 | 28.66 |
| CPU | ONNX INT8 | 4,875 | 10,405 | 418 | 29.07 (+0.41) |
| GPU | PyTorch bf16 | 2,284 | 4,335 | 862 | 28.45 |
| GPU | ONNX FP32 | 1,309 | 2,249 | 1,664 | 28.53 (+0.08) |

Mục tiêu latency < 500 ms **không đạt** ở cấu hình trên (kể cả theo nhóm độ dài input). Benchmark này dùng `strategy: extractive` (so backend trên cùng input). Đường xử lý thật của API (3 mức độ dài như UI, văn bản dài đi hierarchical concat) đo bằng `scripts/benchmark_pipeline.py` (xem Phase 4). INT8 giảm dung lượng trên đĩa (862 → 418 MB) nhưng gần như không giảm RAM đỉnh khi chạy.

**Các thí nghiệm mở rộng** (đều so với ViT5 + LoRA ở trên, cùng tập test):

| Thí nghiệm | Kết quả | Chi tiết |
|---|---|---|
| Văn bản dài: ghép bản tóm tắt chunk thay vì tóm tắt lại (`reduce: concat`) | 225 văn bản dài, ROUGE-L: concat **22.29**, extractive 20.70, extractive ép dài 20.55, tóm tắt lại 13.35 (chunk 128) / 13.48 (chunk 48). Cắt về độ dài của extractive: chưa đủ bằng chứng về khác biệt; so với extractive ép dài: concat hơn có ý nghĩa (xem Giới hạn) | [`results/long_v2/report.md`](results/long_v2/report.md), [`length_control.md`](results/long_v2/length_control.md) |
| Tune `length_penalty` / `min_new_tokens` trên validation | +0.23 ROUGE-L, nằm trong nhiễu → giữ cấu hình cũ | [`results/tune_lengths.md`](results/tune_lengths.md) |
| Distillation: train tiếp trên 4,999 bản tóm tắt do LLM viết | **Kém hơn**: ROUGE-L 24.33 so với 26.98 (−2.65, CI 95% [2.08, 3.21]), kém ở mọi tập con; dài hơn (35.2 so với 26.8 âm tiết) nhưng lệch văn phong reference. Chưa chấm faithfulness | [`results/distill/report.md`](results/distill/report.md) |
| Đối chiếu LLM judge với người chấm (50 mục, chấm mù) | Kappa trọng số 0.70 (CI 95% [0.49, 0.85]), trùng khớp 66%, lệch ≤ 1 điểm 84% | [`results/human_eval/agreement.md`](results/human_eval/agreement.md) |
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
  model_card.py     model card cho Hugging Face Hub, sinh từ results/metrics.csv
  api/              phase 5: app (FastAPI), streaming (SSE), settings
ui/app.py           phase 5: Streamlit, chỉ gọi API qua HTTP
docker/             api.Dockerfile (multi-stage, CPU, non-root) · ui.Dockerfile
demo/space/         demo trên Hugging Face Spaces (Docker: API + UI, model tải từ Hub)
.github/workflows/  CI: test offline (Python 3.11, 3.12) + ruff
tests/              pytest offline (tokenizer / backend giả); test model thật tự skip khi thiếu thư viện hoặc mạng
results/            report.md, qualitative.md, benchmark.md + file số liệu thô
requirements*.txt   requirements.txt (phase 1–3) · -inference (4) · -api (5) · -ui (5) · -ci (CI, không có torch)
```

### Môi trường đã kiểm chứng

| | Train / đánh giá (`.venv`) | Inference ONNX / API (`.venv-onnx`) |
|---|---|---|
| Máy | Windows 11, i7-13620H, RTX 4050 Laptop 6GB | như bên trái |
| Python | 3.13.12 | 3.13 |
| Thư viện chính | torch 2.14.1+cu130, transformers 5.16.1, peft 0.21.2, pandas 2.3.3, rouge-score 0.1.2, underthesea 9.5.0 | transformers 4.57.6, optimum-onnx 0.1.0, onnxruntime 1.23.x (theo `requirements-inference.txt`) |

Cột trái lấy từ `results/long_v2/run_info.json`; từ nay mỗi lần `scripts/evaluate.py` ghi `run_info.json` (commit Git, config, phiên bản thư viện, danh tính model) cạnh kết quả. Hai môi trường tách riêng vì `optimum-onnx` cần transformers < 4.58. CI chạy test offline trên Linux, Python 3.11 và 3.12, không có torch.

## Phase 1 — Dữ liệu

### Dataset

| Dataset | Hub ID | Split | Ghi chú |
|---|---|---|---|
| VietNews (VNDS) | [`nam194/vietnews`](https://huggingface.co/datasets/nam194/vietnews) @ `39e2837` | train 99,134 / val 22,184 / test 22,498 | Văn bản trên Hub đã tách từ (`Hà_Nội`) và tách dấu câu (` , `); pipeline khôi phục văn bản tự nhiên (`desegment: true`). |
| WikiLingua (vi) | [`esdurmus/wiki_lingua`](https://huggingface.co/datasets/esdurmus/wiki_lingua), config `vietnamese` | chỉ có train (6,616 bài, nhiều section/bài) | Tự chia train/val/test 90/5/5 theo `hash(seed, url)`, nên các section của cùng một bài luôn chung một split. |

Schema đầu ra chung: `id, document, summary`, cùng các cột độ dài token và
các cột của chiến lược xử lý văn bản dài.

### Cài đặt & chạy

```bash
make install                      # pip install -r requirements.txt && pip install -e .
export HF_TOKEN=...               # tùy chọn; token chỉ đọc từ biến môi trường
make test                         # pytest
make smoke                        # = python scripts/prepare_data.py --config configs/data.yaml --limit 200
make prepare                      # chạy đầy đủ
```

`--limit N` là tổng số mẫu thô của cả lần chạy, chia đều cho các cặp
(dataset, split) và đọc theo kiểu streaming. Đầu ra gồm
`data/processed/<dataset>/<split>.parquet` và `data/processed/stats.json`.

### Thiết kế chính

- **Chuẩn hóa:** bỏ HTML → Unicode NFC → bỏ ký tự điều khiển/zero-width → (VietNews) khôi phục văn bản bị tách từ → thống nhất dấu thanh cho vần `oa/oe/uy` mở (`tone_style: old` cho ra `hòa, khỏe, thủy`) → gộp khoảng trắng.
- **Tách từ** (underthesea/pyvi) chỉ chạy khi bật `preprocess.word_segmentation.enabled` và ghi ra cột riêng `*_ws` (cho PhoBERT). Input của ViT5 không bao giờ bị tách từ.
- **Lọc:** document hoặc summary rỗng/quá ngắn, summary dài hơn document, trùng lặp trong split và giữa các split (ưu tiên giữ ở test > validation > train).
- **Văn bản dài** (`long_context.strategy`). Mọi độ dài đều đo bằng tokenizer ViT5 và đã trừ chỗ cho `</s>`:
  - `extractive_filter`: chấm điểm câu bằng BM25-centrality hoặc LexRank, chọn câu rồi xuất theo thứ tự gốc, luôn ≤ `max_input_tokens`. Hai chế độ chọn câu (`selection`):
    - `budget` (mặc định): nạp câu theo thứ hạng tới khi đầy ngân sách token.
    - `ratio`: giữ top 30% số câu (baseline).

    Chạy đầy đủ VietNews train với `ratio` cho thấy document bị lọc (trung vị 1,170 token) chỉ còn khoảng 515 token, tức bỏ phí khoảng một nửa ngữ cảnh; vì vậy mặc định được chuyển sang `budget`.
  - `hierarchical`: gom câu thành chunk ≤ `chunk_max_tokens`, overlap khoảng `overlap_tokens` token tính theo câu. Câu nào dài hơn cả một chunk mới bị chia, theo dấu phẩy trước rồi mới tới token.

## Phase 2 — Training

```bash
# Local (CPU) — cài torch CPU trước: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m vnsum.models.train --config configs/train.yaml --max_steps 5 --limit 32      # smoke test
python scripts/merge_lora.py --config configs/train.yaml --adapter outputs/vit5-lora/final_adapter
```

Trên Colab dùng `notebooks/train_colab.ipynb` (T4): checkpoint lưu trên Google Drive, chạy lại cell train sẽ tự resume.

- **LoRA:** r=16, alpha=32, gắn vào `q,k,v,o` của mọi attention (self-attn encoder, self + cross-attn decoder). Bật `lora.include_ffn` để thêm `wi, wo`.
- **Precision:** `configs/train.yaml` (T4) mặc định fp32, vì T4 không có bf16 và T5 dễ NaN ở fp16. `--precision fp16` nhanh hơn; nếu loss NaN/inf, guard dừng ngay và báo lỗi. GPU hỗ trợ bf16 (Ampere/Ada) dùng `precision: bf16`, ví dụ `configs/train_rtx4050.yaml`; config này kế thừa `train.yaml` qua `extends`.
- **Smoke test đã chạy thật (RTX 4050 Laptop):** LoRA trên `q,k,v,o` cho 3,538,944 tham số trainable / tổng 229,489,920 (1.542%). Sau khi merge, logits lệch so với PeftModel tối đa 2.29e-05.
- **Input:** `data.input_strategy` = `extractive_filter` (cột `input_text` của phase 1) hoặc `truncate` (baseline). `hierarchical` dùng lúc suy luận, không dùng để train.
- **ROUGE** trong lúc train tính trên 500 mẫu val ở cuối mỗi epoch, với tokenizer giữ dấu tiếng Việt (tokenizer mặc định của `rouge_score` xóa mọi chữ có dấu).

### Kết quả train (RTX 4050 Laptop 6GB, `configs/train_rtx4050.yaml`)

| | Giá trị |
|---|---|
| Dữ liệu | 116,557 mẫu train (VietNews 99,133 + WikiLingua-vi 17,424), input `extractive_filter[bm25, budget]` |
| Cấu hình | LoRA r=16 trên `q,k,v,o`; bf16; batch 2 × grad accum 16 (batch hiệu dụng 32); LR 5e-4; 1 epoch = 3,643 step |
| Thời gian | 7 giờ 26 phút (~7.35 s/step), VRAM ~2.2 GB |
| Train loss | 4.43 → ~1.85 (trung bình cả epoch 1.938) |
| Eval loss (500 mẫu val) | 2.61 (sau 20 step, lần đo tốc độ) → **1.899** (cuối epoch) |
| ROUGE-1/2/L (500 mẫu val, greedy, tokenizer âm tiết giữ dấu) | 14.88/6.52/12.40 (sau 20 step) → **29.88 / 15.72 / 25.35** (cuối epoch) |
| Merge | max \|Δlogits\| giữa PeftModel và model đã merge = 1.14e-05 |

ROUGE ở bảng trên là số đo sơ bộ trong lúc train: greedy decoding, không chặn lặp từ, chỉ 500 mẫu val. Đánh giá đầy đủ trên tập test ở phase 3 (beam search, `no_repeat_ngram_size`, tách riêng nhóm document > 1024 token).

Ví dụ đầu ra của model đã merge (một bài trong tập test VietNews):
> Sau khi lừa đảo vợ chồng ông Đinh Ngọc H., Huy đã lừa vợ chồng ông Th. 250 triệu đồng.

## Phase 3 — Evaluation

```bash
# Smoke test (CPU, không gọi API trả phí) -> results/smoke/
python scripts/evaluate.py --config configs/eval.yaml --limit 10 --skip_judge --model_path <thư mục model đã merge>
# Đầy đủ, chưa gọi API: tính mọi chỉ số offline + in ước tính số request cho LLM zero-shot và judge
python scripts/evaluate.py --config configs/eval.yaml --model_path <thư mục model đã merge>
# Gọi API trả phí (khóa đọc từ biến môi trường, vd. ANTHROPIC_API_KEY)
python scripts/evaluate.py --config configs/eval.yaml --model_path <...> --allow_api
```

- **Hệ thống:** Lead-3, ViT5 + LoRA (phase 2), `VietAI/vit5-base-vietnews-summarization` (tham chiếu công khai), LLM zero-shot qua API. Các hệ thống chạy trên cùng mẫu test (seed), các model seq2seq dùng chung generation config.
- **ROUGE** dùng tokenizer âm tiết giữ dấu; tokenizer mặc định của `rouge-score` xóa ký tự ngoài ASCII ("Hà Nội" và "Hè Nổi" bị coi là giống nhau, có test chứng minh).
- **BERTScore** được cài đặt trực tiếp theo thuật toán gốc (greedy cosine, không idf), vì thư viện `bert-score` 0.3.13 không cắt input PhoBERT (tokenizer không khai báo `model_max_length`). Điểm là **raw**, vì không có baseline rescale cho PhoBERT.
- **API trả phí** chỉ được gọi khi có `--allow_api`. Dự đoán và kết quả judge được ghi ngay ra cache trong `results/`, nên chạy lại không phải trả phí lần nữa.
- **Khoảng tin cậy:** report có CI 95% bootstrap cho từng chỉ số và so sánh cặp (paired bootstrap) với ViT5 + LoRA trên cùng các mẫu; số liệu thô nằm ở `results/significance.csv`.
- **Kiểm tra judge bằng người chấm** (không gọi API):

  ```bash
  python scripts/human_eval.py export --config configs/eval.yaml    # -> results/human_eval/items.md + annotation_sheet.csv
  # đọc items.md, điền human_score (1–5) vào annotation_sheet.csv
  python scripts/human_eval.py analyze --config configs/eval.yaml   # -> results/human_eval/agreement.md
  ```

  Bộ chấm gồm 50 mục lấy ngẫu nhiên (seed) từ các mẫu judge đã chấm, chia đều cho ViT5 + LoRA, ViT5 công khai và LLM zero-shot, ẩn tên hệ thống và điểm judge. `analyze` báo tỷ lệ trùng khớp, tỷ lệ lệch ≤ 1 điểm, Cohen's kappa trọng số bậc hai (kèm CI bootstrap), Spearman và ma trận nhầm lẫn.

## Phase 4 — Inference optimization

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
- **Đo thêm cấu hình:** `--num_beams 1` (greedy), `--length short`, `--tag <tên>` để ghi ra `results/benchmark_<tên>.md` mà không ghi đè bản chính.
- **FP16 không dùng được:** chuyển FP32 → FP16 (2 lần thử, lần 2 giữ LayerNorm/residual/softmax/FFN đầu ra ở FP32) đều cho chuỗi vô nghĩa, trong khi ONNX FP32 trên cùng GPU cho kết quả đúng (`scripts/check_gpu_onnx.py`). Cờ `--fp16` vẫn còn nhưng không nằm trong benchmark.

### Benchmark đường xử lý của API

```powershell
python scripts/benchmark_pipeline.py --config configs/inference.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
# nhanh: --n_short 5 --n_long 5 ; ONNX: --backend onnx
```

Mô phỏng đúng việc UI làm: 3 request (short, medium, long) cho cùng văn bản, config của API (văn bản > 1024 token đi
hierarchical concat). So hai biến thể trên cùng tài liệu: **cũ** (tóm tắt chunk từng cái, mỗi mức độ dài sinh lại) và
**hiện tại** (`long_document.chunk_batch_size`: sinh chunk theo batch; `chunk_cache_size`: nhớ bản tóm tắt chunk của
văn bản gần nhất, nên mức độ dài thứ 2 và 3 không sinh lại). Báo latency theo chiến lược, tổng thời gian chờ cho 3
mức, theo số chunk, và % bản tóm tắt giống hệt nhau giữa hai biến thể → `results/pipeline_benchmark.md`.

Kết quả (40 văn bản test: 20 ngắn, 20 dài; PyTorch bf16, RTX 4050 Laptop; [`results/pipeline_benchmark.md`](results/pipeline_benchmark.md)):

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

## Phase 5 — API, UI, Docker

### API (FastAPI)

```powershell
# trong .venv-onnx
pip install -r requirements-api.txt
$env:VNSUM_MODEL_PATH = "C:\vnsum-runs\vit5-vnsum-merged"     # bản ONNX lấy ở "<đường dẫn>-onnx"
uvicorn --factory vnsum.api.app:create_app --app-dir src --port 8000
```

| Endpoint | Mô tả |
|---|---|
| `GET /health` | backend đang dùng, các mức độ dài, giới hạn input |
| `POST /summarize` | `{"text": "...", "length": "short\|medium\|long"}` → `summary`, `latency_ms`, `strategy`, số token, `request_id` |
| `POST /summarize/stream` | như trên nhưng trả Server-Sent Events: `start` → nhiều `token` → `done` (hoặc `error`) |

- Model nạp **một lần** khi server khởi động (lifespan). Inference chạy trong threadpool (`run_in_threadpool`) nên handler async không block event loop; số request inference chạy cùng lúc giới hạn bởi `api.max_concurrency` (mặc định 1).
- Validation bằng Pydantic: `text` 1–`api.max_input_chars` ký tự (mặc định 50,000, sau khi bỏ khoảng trắng đầu/cuối), `length` thuộc `lengths` trong config, không nhận trường lạ; sai → 422.
- Mỗi request có request id (header `X-Request-ID` của client, hoặc tự sinh), có trong log, header và body.
- Cấu hình qua biến môi trường (`.env.example`): `VNSUM_MODEL_PATH`, `VNSUM_BACKEND` (onnx | pytorch), `VNSUM_ONNX_VARIANT` (int8 | fp32), `VNSUM_DEVICE`, `VNSUM_STREAM_BACKEND`, `VNSUM_CONFIG`.

**Streaming.** `/summarize/stream` dùng `TextIteratorStreamer` của transformers. Backend ONNX (`ORTModelForSeq2SeqLM`) dùng `generate()` của transformers nên stream được, vì vậy mặc định `api.stream_backend: same` dùng chung model với `/summarize`. Nếu stream với ONNX gặp lỗi trên máy của bạn, đặt `VNSUM_STREAM_BACKEND=pytorch`: API nạp thêm model PyTorch (thư mục `VNSUM_MODEL_PATH`) chỉ để stream. Hai giới hạn cần biết:
- Streamer của transformers không hỗ trợ beam search, nên stream luôn sinh **greedy** (`num_beams=1`, event `done` ghi `"decoding": "greedy"`); kết quả có thể khác `/summarize` (beam 4).
- Văn bản cần hierarchical có nhiều lần sinh: các `token` có `call` 0..k-1 là tóm tắt từng chunk, `call` cuối là bản tóm tắt cuối cùng (cũng có trong `done.summary`).

### UI (Streamlit)

```powershell
pip install -r requirements-ui.txt          # không cần torch / model
$env:API_URL = "http://localhost:8000"
streamlit run ui/app.py
```

Nhập văn bản, bấm **Tóm tắt**: UI gọi `/summarize` ba lần (short, medium, long) và hiển thị ba bản cạnh nhau kèm latency. UI không nạp model.

### Docker

- `docker/api.Dockerfile`: multi-stage (stage 1 cài thư viện vào venv, stage 2 chỉ copy venv + `src/` + `configs/`), torch bản CPU, chạy bằng user không phải root, có `HEALTHCHECK` gọi `/health`.
- `docker/ui.Dockerfile`: chỉ Streamlit + httpx, user không phải root.
- `docker-compose.yml`: `api` (port 8000, mount `${MODEL_DIR}` vào `/models` chỉ đọc) + `ui` (port 8501, chờ `api` healthy). Model không được copy vào image; không có secret trong Dockerfile hay compose.

### Test

```powershell
python -m pytest                                               # mọi test; test thiếu thư viện / model tự bỏ qua
python -m pytest tests/test_api.py tests/test_inference.py     # trong .venv-onnx (cần requirements-api.txt)
ruff check .
```

`tests/test_api.py` dùng `TestClient` với `Summarizer` thật nhưng backend giả, nên không cần model.

**CI** (`.github/workflows/tests.yml`) chạy trên mỗi push / pull request: `pytest` với `requirements-ci.txt` (không
torch / transformers / onnxruntime) trên Python 3.11 và 3.12, và `ruff check` (chỉ lỗi thật: cú pháp, tên chưa
định nghĩa, import thừa). Các test cần thư viện nặng hoặc model thật tự bỏ qua trên CI (`pytest -ra` in lý do) và
cần chạy trên máy có GPU: `tests/test_vit5_tokenizer.py`, phần ONNX của `tests/test_inference.py`, phần train của
`tests/test_training_setup.py`, và test BERTScore / sinh thật trong `tests/test_eval.py`.

## Model trên Hugging Face Hub và demo

```powershell
python scripts/push_to_hub.py --repo_id <tài-khoản>/vit5-vnsum --model_dir C:/vnsum-runs/vit5-vnsum-merged --onnx_dir C:/vnsum-runs/vit5-vnsum-merged-onnx/int8
# chạy thử: chỉ ghi results/model_card.md và liệt kê file. Đọc lại, rồi thêm --push (cần huggingface-cli login).
```

Model card sinh từ `results/metrics.csv` (`src/vnsum/model_card.py`), gồm số đánh giá, cách dùng, giới hạn và giấy
phép dữ liệu. Repo mới tạo ở chế độ **private**; chỉ thêm `--public` khi đã kiểm tra giấy phép của VietNews. Demo trên
Hugging Face Spaces (Docker, CPU): xem [`demo/README.md`](demo/README.md).

## Giới hạn còn lại

- **Mức độ dài:** bản đầu chỉ có giới hạn trên (`max_new_tokens`), nên khi model tự dừng sớm ba mức ra cùng một bản (đã gặp trên UI: 17 token ở cả ba mức). Đã thêm `min_new_tokens` / `length_penalty` cho short và long; đo mức khác biệt bằng `scripts/compare_lengths.py` (→ `results/lengths.md`).
- **Ngoài miền dữ liệu train** (tin tức VietNews, hướng dẫn wikiHow), ví dụ văn bản hành chính dạng danh sách, model thường chỉ chép câu mở đầu — thói quen học từ tin tức, nơi câu đầu thường là tóm tắt tốt.
- Latency < 500 ms chưa đạt ở cấu hình đã đo (beam 4, batch 1); greedy, mức `short` hay batch lớn hơn chưa được benchmark.
- Docker image chỉ có bản CPU; GPU (ONNX FP32 nhanh nhất trong benchmark) mới chạy được ngoài Docker.
- **Văn bản dài:** "long-context" ở đây là khả năng của pipeline chia/lọc văn bản; mỗi lần ViT5 vẫn chỉ nhận ≤ 1,024 token. Mặc định `reduce: concat` (ghép bản tóm tắt chunk, bỏ câu trùng, chọn câu theo ngân sách; `src/vnsum/inference/reduce.py`). Đã đối chứng các yếu tố gây nhiễu mà review chỉ ra, trên 225 văn bản dài (`results/long_v2/`):
  - *Ngân sách chunk:* tóm tắt lại bản ghép cho kết quả như nhau dù chunk 48 hay 128 token (ROUGE-L 13.48 / 13.35, bản cuối ~16 âm tiết), nên thủ phạm là bước tóm tắt lại, không phải ngân sách chunk.
  - *Độ dài:* concat (54.0 âm tiết) hơn extractive (30.8) **+1.59** ROUGE-L, nhưng cắt concat về cùng độ dài thì chỉ còn +0.34 (CI 95% [−0.66, +1.35]): chưa đủ bằng chứng về khác biệt ở độ dài ngắn (không có nghĩa là đã chứng minh hai cách tương đương; cắt sau khi sinh cũng khác với việc hai cách tự viết ngắn từ đầu). Ngược lại, ép extractive viết dài (`vit5_lora_tuned`, min 64 token, 60.7 âm tiết) không giúp gì (ROUGE-L 20.55, ROUGE-1 tăng nhưng ROUGE-2/L không), và concat hơn nó **+1.74** [+0.69, +2.86] dù ngắn hơn; cắt concat về độ dài đó vẫn hơn +1.69 (`scripts/length_control.py` → `results/long_v2/length_control.md`).
  - *Diễn giải (giả thuyết, chưa kiểm chứng):* kết quả gợi ý lợi thế của concat nằm ở chỗ viết được bản tóm tắt dài cỡ reference từ nhiều phần văn bản, còn ép một lần sinh viết dài thì khớp reference kém hơn. ROUGE chỉ đo độ khớp với reference, không đo độ đầy đủ hay tính đúng sự thật; faithfulness của concat chưa được chấm. Cần đánh giá độ đầy đủ và tính trung thực (người chấm hoặc judge) trên cùng các văn bản dài để xác nhận. Concat tốn 2–4 lần sinh. Nhóm dài chỉ có 225/1,910 mẫu.
  - *Chấm người (20 văn bản dài, theo cặp, một người chấm, chấm mù và độc lập với judge; `results/long_v2/human_eval/`):* concat **đầy đủ hơn** (2.70 so với 1.90, +0.80, CI 95% [+0.50, +1.05]) nhưng **kém trung thực hơn** (3.65 so với 4.25, −0.60, CI [−1.25, −0.05], p 0.04): 6/20 bản concat bị chấm 1–2 điểm trung thực so với 3/20 của extractive (vd. sai ngày đăng ký, đổi tên luật sư, sai khóa đại hội). Giả thuyết về độ đầy đủ được xác nhận; đổi lại là nhiều lỗi sự thật hơn, có lẽ vì concat sinh nhiều nội dung hơn từ nhiều chunk. Mẫu nhỏ và một người chấm: mức giảm trung thực sát ngưỡng ý nghĩa, cần người chấm thứ hai. Hướng tiếp theo: kiểm tra sự thật sau khi sinh (đối chiếu tên riêng, ngày, số liệu với văn bản gốc).
- **Kiểm tra sự thật bề mặt (`src/vnsum/factcheck.py`):** đánh dấu câu có số, ngày hoặc tên riêng không tìm thấy trong văn bản gốc (bỏ từ viết hoa đầu câu và chữ cái đơn; chuỗi tên phải tách được thành các tên có trong bài; viết tắt và số La Mã phân biệt hoa thường). Quy tắc được chỉnh trên 40 mục người chấm ở `results/long_v2/human_eval` (khoảng một nửa lỗi trong ghi chú là sai số / tên; nửa còn lại là lỗi quan hệ như gán nhầm lời nói, đảo ý, mà đối chiếu chuỗi không bắt được).
  - *Kiểm chứng độc lập* (`results/factcheck.md`, 800 bản tóm tắt judge đã chấm, không dùng khi chỉnh quy tắc): bản bị đánh dấu có faithfulness TB **2.66** so với **4.28**, tỷ lệ judge ≤ 2 là **58%** so với **14%** (gấp ~4 lần); nhưng chỉ phủ **21%** số bản judge chấm ≤ 2. Bộ 50 mục người chấm ở `results/human_eval` (cũng độc lập): 3.40 so với 4.51 (5 mục bị đánh dấu). Tỷ lệ bị đánh dấu trên 1,910 mẫu test: Lead-3 0.0% (chép câu nên không thể thêm số / tên — kiểm tra hợp lý), ViT5 + LoRA 4.9%, LLM zero-shot 4.5%, ViT5 công khai 12.5%, reference 25.4% (bản tóm tắt gốc hay nêu thông tin không có trong thân bài, nên con số là mức nền, không phải tỷ lệ bịa).
  - *Bỏ câu bị đánh dấu trong concat* (`results/long_fc`, 225 văn bản dài): đổi 16 văn bản; ROUGE-L −0.03 (CI 95% [−0.28, +0.19]), BERTScore −0.07, tỷ lệ bị đánh dấu 6.7% → 0%. **Bật mặc định** cho API (`long_document.factcheck: drop`) vì sai số / tên gây hại hơn mất một câu; nhưng mức cải thiện faithfulness của việc bỏ câu **chưa đo trực tiếp** (cần judge hoặc người chấm), và có đánh dấu nhầm (bài viết tắt "Bộ GTVT" còn bản tóm tắt viết đầy đủ → mất một câu đúng). Ví dụ bắt đúng: "xã Tóc Tiên, huyện Hóc Môn, TP. HCM" (Tóc Tiên không thuộc Hóc Môn).
  - Cũng dùng làm chỉ số `unsupported_pct` trong báo cáo đánh giá (`aux_metrics.factcheck`) và cảnh báo trên API / UI (`factcheck_warnings`, trường `unsupported`; không sửa bản tóm tắt).
- **Bỏ câu trùng (đã sửa):** bản đầu chỉ so tỷ lệ bigram nên gộp nhầm câu khác số liệu ("doanh thu 2023 đạt 100 tỷ" / "2024 đạt 200 tỷ") hoặc khác phủ định ("đã" / "chưa xác nhận"). Review lần 2 tìm thêm hai trường hợp: tên riêng đứng đầu câu ("Nam" / "Lan đã nhận học bổng...") và cùng tập số nhưng đảo quan hệ ("2023 đạt 100, 2024 đạt 200" / "2023 đạt 200, 2024 đạt 100"). Nay câu chỉ bị bỏ khi mọi từ viết hoa của nó (kể cả đầu câu) có trong câu được giữ, các số xuất hiện theo **cùng thứ tự**, và hai câu cùng phủ định; không chắc thì giữ cả hai (`tests/test_reduce.py`). Vẫn chỉ là xấp xỉ: không phân tích được quan hệ cú pháp. Đo lại sau mỗi lần sửa: ROUGE-L 22.25 → 22.26 → 22.29, không làm giảm chất lượng.
- **Human eval:** một người chấm, 50 mục, kappa 0.70 là kiểm tra ban đầu, chưa đủ để khẳng định judge đáng tin trong mọi trường hợp. Đối chiếu lại với văn bản gốc, 3/8 mục bất đồng nặng là lỗi sự thật mà người chấm bỏ sót (H020 gán nhầm độ cao 10.000 m cho chuyến QF94; H038 gán sai ngày 25/8/2017; H027 thêm chức danh không có trong bài), 2/8 là judge quá khắt khe (H004, H033), 3/8 cả hai cùng lệch. Điểm người chấm được giữ nguyên (đã chấm mù); cần thêm người chấm độc lập, và đánh giá thêm độ đầy đủ và mạch lạc. Để chấm độc lập concat và extractive **trước** judge: `python scripts/human_eval.py export --config configs/eval.yaml --results_dir results/long_v2 --systems vit5_lora,vit5_lora_hier_concat --n_items 40 --paired --unjudged --coverage` (cùng văn bản cho hai hệ thống, thêm điểm độ đầy đủ; `analyze` báo điểm người và so sánh cặp, đối chiếu judge nếu có).
- **ViT5 + LoRA thua ViT5 công khai riêng trên VietNews** (ROUGE-L 26.75 so với 32.11); chỉ hơn trên tập hỗn hợp nhờ WikiLingua. Không kiểm chứng được model công khai có train trên tập test VietNews hay không.
- **Cache dự đoán (đã sửa):** trước đây khóa cache chỉ gồm config + đường dẫn model + id mẫu, nên sửa dữ liệu (giữ id) hoặc ghi đè trọng số trong cùng thư mục vẫn trả dự đoán cũ. Nay cache kiểm tra thêm hash input của từng mẫu và hash trọng số model; mỗi lần đánh giá ghi `run_info.json` (commit Git, config, phiên bản thư viện, danh tính model). Dự đoán LLM cũ (trả phí) chưa có hash input vẫn được dùng lại, kèm cảnh báo. Review lần 3: khóa cache từng bỏ qua `batch_size` chung (đổi batch vẫn trả output của batch cũ), trong khi batch làm đổi output ở ~26% văn bản dài; nay khóa gồm cấu hình sinh thực tế sau override, kể cả `batch_size`.
- API không có xác thực hay giới hạn tần suất request: chỉ dùng cho demo / mạng nội bộ.

## Giấy phép dữ liệu

WikiLingua: CC BY 3.0 (nội dung wikiHow). VietNews: xem bài báo VNDS (Nguyen et al., NICS 2019) và repo gốc `ThanhChinhBK/vietnews`.
