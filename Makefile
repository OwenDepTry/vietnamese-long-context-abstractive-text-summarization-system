PYTHON ?= python
CONFIG ?= configs/data.yaml
LIMIT  ?= 200

.PHONY: help install test smoke prepare train-smoke merge

help:
	@echo "make install  - cài dependency (requirements.txt) + package vnsum (editable)"
	@echo "make test     - chạy pytest"
	@echo "make smoke    - chạy pipeline dữ liệu với $(LIMIT) mẫu (CPU)"
	@echo "make prepare  - chạy pipeline dữ liệu đầy đủ"
	@echo "make train-smoke - phase 2: 5 step LoRA trên CPU với 32 mẫu"
	@echo "make merge    - phase 2: merge adapter vào model đầy đủ"

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
