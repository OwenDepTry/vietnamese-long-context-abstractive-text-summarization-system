# syntax=docker/dockerfile:1
# UI Streamlit: chỉ gọi API qua HTTP, không có model / torch trong image.
FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY ui/ ui/
ENV API_URL=http://api:8000
USER app
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
