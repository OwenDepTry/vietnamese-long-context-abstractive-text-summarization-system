# syntax=docker/dockerfile:1
# Demo vnsum trên Hugging Face Spaces (Docker, CPU): API + UI trong một container, model tải từ Hub lúc khởi động.
# Code lấy từ GitHub tại lúc build (đổi VNSUM_REF để cố định một commit).
FROM python:3.13-slim
ARG VNSUM_REPO=https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system.git
ARG VNSUM_REF=main
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Spaces chạy container với uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app
RUN git clone --depth 1 --branch "${VNSUM_REF}" "${VNSUM_REPO}" /app && chown -R user /app
# torch bản CPU (cùng version với docker/api.Dockerfile), rồi thư viện API + UI.
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt -r requirements-ui.txt
ENV PYTHONPATH=/app/src VNSUM_CONFIG=configs/inference.yaml HOME=/home/user
USER user
EXPOSE 7860
CMD ["python", "demo/space/start.py"]
