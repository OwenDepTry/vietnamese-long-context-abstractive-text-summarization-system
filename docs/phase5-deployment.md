# Phase 5 — API, UI, Docker

← [Về README](../README.md)

## API (FastAPI)

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

## UI (Streamlit)

```powershell
pip install -r requirements-ui.txt          # không cần torch / model
$env:API_URL = "http://localhost:8000"
streamlit run ui/app.py
```

Nhập văn bản, bấm **Tóm tắt**: UI gọi `/summarize` ba lần (short, medium, long) và hiển thị ba bản cạnh nhau kèm latency. UI không nạp model.

## Docker

- `docker/api.Dockerfile`: multi-stage (stage 1 cài thư viện vào venv, stage 2 chỉ copy venv + `src/` + `configs/`), torch bản CPU, chạy bằng user không phải root, có `HEALTHCHECK` gọi `/health`.
- `docker/ui.Dockerfile`: chỉ Streamlit + httpx, user không phải root.
- `docker-compose.yml`: `api` (port 8000, mount `${MODEL_DIR}` vào `/models` chỉ đọc) + `ui` (port 8501, chờ `api` healthy). Model không được copy vào image; không có secret trong Dockerfile hay compose.

## Test

```powershell
python -m pytest                                               # mọi test; test thiếu thư viện / model tự bỏ qua
python -m pytest tests/test_api.py tests/test_inference.py     # trong .venv-onnx (cần requirements-api.txt)
ruff check .
```

**Pre-commit hook** (`.pre-commit-config.yaml`): `make hooks` cài một lần; sau đó mỗi `git commit` tự chạy `ruff`,
xóa output notebook (`nbstripout`), chặn file còn dấu conflict, file lớn hơn 10 MB và file lẫn CRLF. Chạy tay trên mọi
file: `pre-commit run --all-files`. Xuống dòng được cố định là LF trên mọi máy bằng `.gitattributes`.

`tests/test_api.py` dùng `TestClient` với `Summarizer` thật nhưng backend giả, nên không cần model.

**CI** (`.github/workflows/tests.yml`) chạy trên mỗi push / pull request: `pytest` với `requirements-ci.txt` (không
torch / transformers / onnxruntime) trên Python 3.11 và 3.12, và `ruff check` (chỉ lỗi thật: cú pháp, tên chưa
định nghĩa, import thừa). Các test cần thư viện nặng hoặc model thật tự bỏ qua trên CI (`pytest -ra` in lý do) và
cần chạy trên máy có GPU: `tests/test_vit5_tokenizer.py`, phần ONNX của `tests/test_inference.py`, phần train của
`tests/test_training_setup.py`, và test BERTScore / sinh thật trong `tests/test_eval.py`.
