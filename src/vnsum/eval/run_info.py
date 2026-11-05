"""Ghi kèm mỗi lần đánh giá: commit Git, config đã resolve, phiên bản thư viện, danh tính model.

Để một bảng số trong ``results/`` luôn truy ngược được về đúng code, config và checkpoint đã tạo ra nó.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "transformers", "peft", "onnxruntime", "optimum", "pandas", "rouge-score", "underthesea")


def git_state(cwd: str | Path | None = None) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        # encoding="utf-8": mặc định Windows đọc bằng cp1252 và vỡ khi diff có chữ tiếng Việt. Thông tin phụ này
        # không bao giờ được làm hỏng lần đánh giá: lỗi gì cũng trả None.
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=30, check=True).stdout
            return out.strip() if out is not None else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain", "--untracked-files=no")
    state: dict[str, Any] = {"commit": commit, "dirty": bool(status) if status is not None else None}
    if state["dirty"]:
        # Commit hash chưa đủ xác định code đã chạy khi còn thay đổi chưa commit -> giữ lại chính diff đó.
        state["diff"] = _run("diff", "HEAD")
    return state


def package_versions(names=PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def write_run_info(path: str | Path, *, argv: list[str], cfg: dict[str, Any], systems: dict[str, Any],
                   n_samples: int) -> Path:
    path = Path(path)
    try:
        git = git_state(Path(__file__).resolve().parents[3])
    except Exception as err:  # noqa: BLE001 - không để metadata làm hỏng lần đánh giá
        git = {"error": f"{type(err).__name__}: {err}"}
    diff = git.pop("diff", None)
    if diff:
        diff_path = path.with_name(path.stem + ".diff")
        diff_path.write_text(diff + "\n", encoding="utf-8")
        git["diff_file"] = diff_path.name
        git["diff_sha1"] = hashlib.sha1(diff.encode("utf-8")).hexdigest()[:12]
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": argv,
        "git": git,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "n_samples": n_samples,
        "systems": systems,
        "config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
