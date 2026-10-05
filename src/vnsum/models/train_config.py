"""Đọc, kiểm tra và áp dụng override cho ``configs/train.yaml`` (không import torch)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PRECISIONS = ("fp32", "fp16", "bf16")  # bf16 chỉ trên GPU hỗ trợ (Ampere/Ada); T4 thì không
INPUT_STRATEGIES = ("extractive_filter", "truncate")
_REQUIRED = ("seed", "paths", "hf", "model", "data", "eval", "lora", "precision", "resume", "training")


class TrainConfigError(ValueError):
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Trộn sâu ``override`` vào bản sao của ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def validate_train_config(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise TrainConfigError(f"Thiếu khóa config: {missing}")
    precision = cfg["precision"]
    if precision not in PRECISIONS:
        raise TrainConfigError(f"precision={precision!r} không hợp lệ; chọn {PRECISIONS}")
    strategy = cfg["data"]["input_strategy"]
    if strategy == "hierarchical":
        raise TrainConfigError(
            "data.input_strategy=hierarchical không dùng để train: không có nhãn tóm tắt cho từng chunk. "
            "Train bằng extractive_filter hoặc truncate; hierarchical áp dụng lúc suy luận (phase 3/4) "
            "với chính model này."
        )
    if strategy not in INPUT_STRATEGIES:
        raise TrainConfigError(f"data.input_strategy={strategy!r} không hợp lệ; chọn {INPUT_STRATEGIES}")
    for key in ("max_source_length", "max_target_length"):
        if int(cfg["data"][key]) <= 0:
            raise TrainConfigError(f"data.{key} phải > 0")
    if not cfg["data"].get("datasets"):
        raise TrainConfigError("data.datasets rỗng")
    lora = cfg["lora"]
    if int(lora["r"]) <= 0 or not lora.get("target_modules"):
        raise TrainConfigError("lora.r phải > 0 và lora.target_modules không được rỗng")
    for banned in ("output_dir", "fp16", "bf16", "seed", "predict_with_generate"):
        if banned in cfg["training"]:
            raise TrainConfigError(
                f"training.{banned} được đặt tự động (paths.output_dir / precision / seed); xóa khỏi mục training."
            )


def _read_yaml_with_extends(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Đọc YAML; nếu có ``extends: <file>`` (tương đối với file hiện tại) thì trộn sâu lên file gốc."""
    path = path.resolve()
    if path in seen:
        raise TrainConfigError(f"Vòng lặp extends: {' -> '.join(map(str, seen + (path,)))}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TrainConfigError(f"{path} không phải mapping YAML")
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(_read_yaml_with_extends(path.parent / parent, seen + (path,)), cfg)
    return cfg


def load_train_config(path: str | Path) -> dict[str, Any]:
    cfg = _read_yaml_with_extends(Path(path))
    validate_train_config(cfg)
    return cfg


def apply_cli_overrides(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    max_steps: int | None = None,
    output_dir: str | None = None,
    processed_dir: str | None = None,
    precision: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    """Áp dụng override từ CLI. ``--limit`` bật profile ``smoke`` trong config."""
    out = copy.deepcopy(cfg)
    if limit is not None:
        if limit <= 0:
            raise TrainConfigError("--limit phải > 0")
        smoke = out.pop("smoke", {}) or {}
        out = deep_merge(out, smoke)
        out["limit"] = int(limit)
    else:
        out.pop("smoke", None)
        out["limit"] = None
    if max_steps is not None:
        out["training"]["max_steps"] = int(max_steps)
    if output_dir:
        out["paths"]["output_dir"] = output_dir
    if processed_dir:
        out["paths"]["processed_dir"] = processed_dir
    if precision:
        out["precision"] = precision
    if resume:
        out["resume"] = resume
    validate_train_config(out)
    return out
