"""Cấu hình API: đường dẫn và backend đọc từ biến môi trường, giới hạn đọc từ mục ``api`` của
``configs/inference.yaml``. Không có secret nào ở đây (model nằm ngoài repo, mount qua volume)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_API_CONFIG: dict[str, Any] = {
    "max_input_chars": 50000,   # giới hạn độ dài text của request (ký tự)
    "max_concurrency": 1,       # số request inference chạy cùng lúc (CPU-bound: 1 tránh tranh luồng)
    "stream_backend": "same",   # same | pytorch — backend cho /summarize/stream
}
STREAM_BACKENDS = ("same", "pytorch")


@dataclass
class ApiSettings:
    config_path: str = "configs/inference.yaml"
    model_path: str | None = None        # thư mục model PyTorch đã merge; ONNX nằm ở "<model_path>-onnx"
    backend: str | None = None           # pytorch | onnx (None = theo config)
    onnx_variant: str | None = None      # fp32 | int8 (None = theo config)
    device: str | None = None            # auto | cpu | cuda
    stream_backend: str | None = None    # same | pytorch (None = theo config)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ApiSettings":
        env = os.environ.get
        return cls(
            config_path=env("VNSUM_CONFIG", "configs/inference.yaml"),
            model_path=env("VNSUM_MODEL_PATH") or None,
            backend=env("VNSUM_BACKEND") or None,
            onnx_variant=env("VNSUM_ONNX_VARIANT") or None,
            device=env("VNSUM_DEVICE") or None,
            stream_backend=env("VNSUM_STREAM_BACKEND") or None,
            log_level=env("VNSUM_LOG_LEVEL", "INFO"),
        )


def api_config(cfg: dict[str, Any], settings: ApiSettings) -> dict[str, Any]:
    out = {**DEFAULT_API_CONFIG, **(cfg.get("api") or {})}
    if settings.stream_backend:
        out["stream_backend"] = settings.stream_backend
    if out["stream_backend"] not in STREAM_BACKENDS:
        raise ValueError(f"api.stream_backend phải thuộc {STREAM_BACKENDS}")
    if int(out["max_input_chars"]) <= 0 or int(out["max_concurrency"]) <= 0:
        raise ValueError("api.max_input_chars và api.max_concurrency phải > 0")
    return out
