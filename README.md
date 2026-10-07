# vnsum — Tóm tắt trừu tượng văn bản tiếng Việt dài

ViT5-base (`VietAI/vit5-base`) fine-tune bằng LoRA trên VietNews + WikiLingua (tiếng Việt), xử lý văn bản dài hơn ngữ cảnh 1024 token, đánh giá bằng ROUGE giữ dấu / BERTScore PhoBERT / LLM-as-Judge, tối ưu inference bằng ONNX Runtime và đóng gói thành API + UI chạy bằng Docker.

| Phase | Nội dung | Kết quả |
|---|---|---|
| 1 | Dữ liệu: tải, chuẩn hóa, lọc, xử lý văn bản dài | `data/processed/*.parquet` |
| 2 | Training: ViT5 + LoRA, merge | model đã merge (ngoài repo) |
| 3 | Evaluation: 4 hệ thống trên 1,910 mẫu test | [`results/report.md`](results/report.md), [`results/qualitative.md`](results/qualitative.md) |
| 4 | Inference: ONNX (KV cache) + INT8, benchmark CPU/GPU | [`results/benchmark.md`](results/benchmark.md) |
| 5 | API (FastAPI) + UI (Streamlit) + Docker | `src/vnsum/api/`, `ui/`, `docker-compose.yml` |

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

Mục tiêu latency < 500 ms **không đạt** ở cấu hình trên (kể cả theo nhóm độ dài input).

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
scripts/            prepare_data · merge_lora · evaluate · human_eval · export_onnx · benchmark · profile_onnx · check_gpu_onnx · summarize
src/vnsum/
  data/             phase 1: load, preprocess (NFC, dấu thanh, tách câu, lọc, dedup), long_context (BM25/LexRank, chunk), tokenization
  models/           phase 2: train (Seq2SeqTrainer + LoRA), lora, callbacks, rouge (tokenizer giữ dấu)
  eval/             phase 3: systems, metrics, bertscore, judge, llm_client, stats (bootstrap), report, human_eval
  inference/        phase 4: predictor (Summarizer), export (ONNX/INT8/FP16), tokenizer, bench, hardware
  api/              phase 5: app (FastAPI), streaming (SSE), settings
ui/app.py           phase 5: Streamlit, chỉ gọi API qua HTTP
docker/             api.Dockerfile (multi-stage, CPU, non-root) · ui.Dockerfile
tests/              pytest offline (tokenizer / backend giả); test model thật tự skip khi thiếu thư viện hoặc mạng
results/            report.md, qualitative.md, benchmark.md + file số liệu thô
requirements*.txt   requirements.txt (phase 1–3) · -inference (4) · -api (5) · -ui (5)
```

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
- **`Summarizer`** (`src/vnsum/inference/predictor.py`): một interface cho backend PyTorch và ONNX, ba mức độ dài `short/medium/long` → `max_new_tokens` 48/128/256, văn bản dài xử lý theo `long_document.strategy` (đưa thẳng / extractive filter như lúc train / hierarchical: tóm tắt từng chunk rồi tóm tắt bản ghép).
- **Benchmark:** cùng 100 tài liệu test và generation config với phase 3, 5 lần warmup, mỗi lượt chạy một process riêng; đo p50/p95, throughput, RAM/VRAM đỉnh, kích thước trên đĩa và ROUGE-L so với PyTorch (cảnh báo khi giảm quá 1 điểm); ghi cấu hình phần cứng thật.
- **Spin-wait của ONNX Runtime** được tắt (`onnx.allow_spinning: false`): khi bật, thread pool của ORT giành CPU với PyTorch trong vòng beam search (đo bằng `scripts/profile_onnx.py`: 3,829 → 1,683 ms trên 2 tài liệu).
- **FP16 không dùng được:** chuyển FP32 → FP16 (2 lần thử, lần 2 giữ LayerNorm/residual/softmax/FFN đầu ra ở FP32) đều cho chuỗi vô nghĩa, trong khi ONNX FP32 trên cùng GPU cho kết quả đúng (`scripts/check_gpu_onnx.py`). Cờ `--fp16` vẫn còn nhưng không nằm trong benchmark.

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
python -m pytest tests/test_api.py tests/test_inference.py     # trong .venv-onnx (cần requirements-api.txt)
```

`tests/test_api.py` dùng `TestClient` với `Summarizer` thật nhưng backend giả, nên không cần model.

## Giới hạn còn lại

- **Mức độ dài là giới hạn trên** (`max_new_tokens` 48/128/256), không ép độ dài tối thiểu: khi model tự kết thúc sớm, ba mức cho cùng một bản tóm tắt (đã gặp khi chạy thử UI với một văn bản quy định 17 token ở cả ba mức).
- **Ngoài miền dữ liệu train** (tin tức VietNews, hướng dẫn wikiHow), ví dụ văn bản hành chính dạng danh sách, model thường chỉ chép câu mở đầu — thói quen học từ tin tức, nơi câu đầu thường là tóm tắt tốt.
- Latency < 500 ms chưa đạt ở cấu hình đã đo (beam 4, batch 1); greedy, mức `short` hay batch lớn hơn chưa được benchmark.
- Docker image chỉ có bản CPU; GPU (ONNX FP32 nhanh nhất trong benchmark) mới chạy được ngoài Docker.
- Pipeline hierarchical mới được kiểm tra bằng test với backend giả, chưa đánh giá chất lượng trên model thật.
- Faithfulness của judge chưa được đối chiếu với người chấm (công cụ `scripts/human_eval.py` đã có).
- API không có xác thực hay giới hạn tần suất request: chỉ dùng cho demo / mạng nội bộ.

## Giấy phép dữ liệu

WikiLingua: CC BY 3.0 (nội dung wikiHow). VietNews: xem bài báo VNDS (Nguyen et al., NICS 2019) và repo gốc `ThanhChinhBK/vietnews`.
