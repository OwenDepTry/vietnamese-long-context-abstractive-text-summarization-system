"""Điểm vào của demo trên Hugging Face Spaces: tải model từ Hub, chạy API (nội bộ) rồi UI Streamlit (cổng 7860).

Biến môi trường (đặt trong Settings của Space):
  VNSUM_HF_REPO   repo model trên Hub, vd. ten-tai-khoan/vit5-vnsum (bắt buộc)
  HF_TOKEN        (Secret) chỉ cần khi repo model là private
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_PORT = 8000
UI_PORT = int(os.environ.get("PORT", "7860"))


def prepare_model(repo_id: str, models_dir: Path, download=None) -> tuple[Path, bool]:
    """Tải model về ``models_dir/vit5-vnsum-merged`` và (nếu có) ONNX INT8 về ``vit5-vnsum-merged-onnx/int8``,
    đúng bố cục mà ``vnsum.inference.config.onnx_dirs`` đọc. Trả (thư mục model, có ONNX hay không)."""
    if download is None:
        from huggingface_hub import snapshot_download as download
    model_dir = models_dir / "vit5-vnsum-merged"
    download(repo_id, local_dir=str(model_dir), ignore_patterns=["onnx/*", "*.md", ".gitattributes"])
    staging = models_dir / "_onnx_download"
    download(repo_id, local_dir=str(staging), allow_patterns=["onnx/int8/*"])
    src = staging / "onnx" / "int8"
    has_onnx = src.is_dir() and any(src.glob("*.onnx"))
    if has_onnx:
        dst = models_dir / "vit5-vnsum-merged-onnx" / "int8"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    shutil.rmtree(staging, ignore_errors=True)
    return model_dir, has_onnx


def api_env(model_dir: Path, has_onnx: bool, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    env.update({
        "VNSUM_MODEL_PATH": str(model_dir),
        "VNSUM_BACKEND": "onnx" if has_onnx else "pytorch",
        "VNSUM_ONNX_VARIANT": "int8",
        "VNSUM_DEVICE": "cpu",
        "HF_HUB_OFFLINE": "1",
    })
    return env


def wait_for_api(url: str, timeout_s: float = 900, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"API dừng khi đang khởi động (mã {proc.returncode}); xem log phía trên.")
        try:
            if urllib.request.urlopen(url, timeout=5).status == 200:
                return
        except OSError:
            pass
        time.sleep(3)
    raise SystemExit(f"API không sẵn sàng sau {timeout_s:.0f} s")


def main() -> int:
    repo_id = os.environ.get("VNSUM_HF_REPO")
    if not repo_id:
        raise SystemExit("Chưa đặt VNSUM_HF_REPO (Settings của Space → Variables).")
    models_dir = Path(os.environ.get("VNSUM_MODELS_DIR", Path.home() / "models"))
    print(f"Tải model {repo_id} ...", flush=True)
    model_dir, has_onnx = prepare_model(repo_id, models_dir)
    print(f"Model: {model_dir} (backend {'ONNX INT8' if has_onnx else 'PyTorch'})", flush=True)

    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--factory", "vnsum.api.app:create_app",
         "--host", "127.0.0.1", "--port", str(API_PORT)],
        env=api_env(model_dir, has_onnx),
    )
    wait_for_api(f"http://127.0.0.1:{API_PORT}/health", proc=api)
    ui_env = dict(os.environ, API_URL=f"http://127.0.0.1:{API_PORT}")
    ui = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0",
         f"--server.port={UI_PORT}", "--server.headless=true", "--browser.gatherUsageStats=false"],
        env=ui_env,
    )
    # Một trong hai dừng thì dừng cả container (Space sẽ báo lỗi thay vì chạy nửa vời).
    while api.poll() is None and ui.poll() is None:
        time.sleep(5)
    for proc in (api, ui):
        if proc.poll() is None:
            proc.terminate()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
