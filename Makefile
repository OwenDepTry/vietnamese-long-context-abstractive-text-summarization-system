PYTHON ?= python
CONFIG ?= configs/data.yaml
LIMIT  ?= 200

.PHONY: help install test smoke prepare

help:
	@echo "make install  - cài dependency (requirements.txt) + package vnsum (editable)"
	@echo "make test     - chạy pytest"
	@echo "make smoke    - chạy pipeline dữ liệu với $(LIMIT) mẫu (CPU)"
	@echo "make prepare  - chạy pipeline dữ liệu đầy đủ"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --limit $(LIMIT)

prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)
