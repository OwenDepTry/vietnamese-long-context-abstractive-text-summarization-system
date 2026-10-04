"""Đọc và kiểm tra file cấu hình YAML.

Mọi module của phase 1 nhận config dưới dạng ``dict`` lấy từ đây, để không có
hyperparameter hay đường dẫn nào bị hardcode trong code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_TOP_LEVEL = (
    "seed",
    "paths",
    "hf",
    "tokenizer",
    "datasets",
    "loading",
    "preprocess",
    "filter",
    "sentence_split",
    "long_context",
    "stats",
)

_ENUMS = {
    ("preprocess", "tone_style"): {"old", "new", None},
    ("preprocess", "word_segmentation", "backend"): {"underthesea", "pyvi"},
    ("filter", "length_unit"): {"words", "chars"},
    ("filter", "dedup", "key"): {"document", "document_summary"},
    ("sentence_split", "backend"): {"rule", "underthesea"},
    ("long_context", "strategy"): {"extractive_filter", "hierarchical"},
    ("long_context", "extractive_filter", "ranker"): {"bm25", "lexrank"},
    ("long_context", "extractive_filter", "selection"): {"budget", "ratio"},
}


class ConfigError(ValueError):
    """Config thiếu khóa hoặc có giá trị không hợp lệ."""


def _get(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Thiếu khóa config: {'.'.join(path)}")
        node = node[key]
    return node


def validate_config(cfg: dict[str, Any]) -> None:
    """Kiểm tra các khóa bắt buộc và giá trị enum; raise ``ConfigError`` nếu sai."""
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ConfigError(f"Thiếu khóa config cấp cao nhất: {missing}")

    for path, allowed in _ENUMS.items():
        value = _get(cfg, path)
        if value not in allowed:
            raise ConfigError(f"{'.'.join(path)}={value!r} không hợp lệ; chọn một trong {sorted(map(str, allowed))}")

    lc = cfg["long_context"]
    reserve = int(_get(cfg, ("tokenizer", "reserve_special_tokens")))
    if int(lc["max_input_tokens"]) <= reserve:
        raise ConfigError("long_context.max_input_tokens phải lớn hơn tokenizer.reserve_special_tokens")
    ratio = float(_get(cfg, ("long_context", "extractive_filter", "keep_ratio")))
    if not 0.0 < ratio <= 1.0:
        raise ConfigError("long_context.extractive_filter.keep_ratio phải trong (0, 1]")
    hier = lc["hierarchical"]
    if int(hier["chunk_max_tokens"]) <= reserve:
        raise ConfigError("long_context.hierarchical.chunk_max_tokens quá nhỏ")
    if not 0 <= int(hier["overlap_tokens"]) < int(hier["chunk_max_tokens"]):
        raise ConfigError("long_context.hierarchical.overlap_tokens phải trong [0, chunk_max_tokens)")

    for name, ds in cfg["datasets"].items():
        if not ds.get("enabled", False):
            continue
        if "hf_id" not in ds:
            raise ConfigError(f"datasets.{name}.hf_id bị thiếu")
        ratios = ds.get("split_ratios")
        if ratios is not None:
            total = sum(float(v) for v in ratios.values())
            if abs(total - 1.0) > 1e-6:
                raise ConfigError(f"datasets.{name}.split_ratios phải có tổng = 1 (hiện {total})")


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc YAML và kiểm tra hợp lệ."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} không phải một mapping YAML")
    validate_config(cfg)
    return cfg
