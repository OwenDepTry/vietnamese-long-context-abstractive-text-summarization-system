"""Ghi kèm mỗi lần đánh giá: commit Git, config đã resolve, phiên bản thư viện, danh tính model.

Để một bảng số trong ``results/`` luôn truy ngược được về đúng code, config và checkpoint đã tạo ra nó.
"""

from __future__ import annotations

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
        try:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


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
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": argv,
        "git": git_state(Path(__file__).resolve().parents[3]),
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
