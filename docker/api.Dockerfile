# syntax=docker/dockerfile:1
# API vnsum (CPU). Model KHÔNG nằm trong image: mount thư mục chứa model vào /models (xem docker-compose.yml).

# ---- stage 1: cài thư viện vào venv ----
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# torch bản CPU (không kéo CUDA ~ vài GB); cùng version với phase 4.
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-api.txt

# ---- stage 2: runtime ----
FROM python:3.13-slim
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src/ src/
COPY configs/ configs/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    VNSUM_CONFIG=configs/inference.yaml \
    VNSUM_MODEL_PATH=/models/vit5-vnsum-merged \
    VNSUM_BACKEND=onnx \
    VNSUM_ONNX_VARIANT=int8 \
    VNSUM_DEVICE=cpu
USER app
EXPOSE 8000
# Nạp model mất vài chục giây -> start-period dài.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
CMD ["uvicorn", "--factory", "vnsum.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
