PYTHON ?= python
CONFIG ?= configs/data.yaml
LIMIT  ?= 200

.PHONY: help install test smoke prepare train-smoke merge eval-smoke eval api ui docker-build docker-up

help:
	@echo "make install  - cài dependency (requirements.txt) + package vnsum (editable)"
	@echo "make test     - chạy pytest"
	@echo "make smoke    - chạy pipeline dữ liệu với $(LIMIT) mẫu (CPU)"
	@echo "make prepare  - chạy pipeline dữ liệu đầy đủ"
	@echo "make train-smoke - phase 2: 5 step LoRA trên CPU với 32 mẫu"
	@echo "make merge    - phase 2: merge adapter vào model đầy đủ"
	@echo "make eval-smoke MODEL=<dir> - phase 3: 10 mẫu, không judge"
	@echo "make eval MODEL=<dir>  - phase 3: đầy đủ, KHÔNG gọi API (in ước tính); thêm --allow_api thủ công"
	@echo "make api      - phase 5: chạy API (cần VNSUM_MODEL_PATH, trong .venv-onnx)"
	@echo "make ui       - phase 5: chạy UI Streamlit (API_URL, mặc định http://localhost:8000)"
	@echo "make docker-build / docker-up - phase 5: docker compose build / up"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --limit $(LIMIT)

prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

train-smoke:
	$(PYTHON) -m vnsum.models.train --config configs/train.yaml --max_steps 5 --limit 32

merge:
	$(PYTHON) scripts/merge_lora.py --config configs/train.yaml

MODEL ?=
MODEL_ARG = $(if $(MODEL),--model_path $(MODEL),)

eval-smoke:
	$(PYTHON) scripts/evaluate.py --config configs/eval.yaml --limit 10 --skip_judge $(MODEL_ARG)

eval:
	$(PYTHON) scripts/evaluate.py --config configs/eval.yaml $(MODEL_ARG)

api:
	$(PYTHON) -m uvicorn --factory vnsum.api.app:create_app --app-dir src --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m streamlit run ui/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up
