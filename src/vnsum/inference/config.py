"""Đọc và kiểm tra ``configs/inference.yaml`` (không import torch / onnxruntime)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BACKENDS = ("pytorch", "onnx")
ONNX_VARIANTS = ("fp32", "int8", "fp16")
STRATEGIES = ("auto", "extractive", "hierarchical")
QUANT_ISAS = ("avx2", "avx512", "avx512_vnni", "arm64")
_REQUIRED = ("model", "backend", "pytorch", "onnx", "export", "generation", "lengths", "default_length",
             "data_config", "max_input_tokens", "long_document", "benchmark")


class InferenceConfigError(ValueError):
    pass


def validate_inference_config(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise InferenceConfigError(f"Thiếu khóa config: {missing}")
    if cfg["backend"] not in BACKENDS:
        raise InferenceConfigError(f"backend phải thuộc {BACKENDS}")
    if cfg["onnx"]["variant"] not in ONNX_VARIANTS:
        raise InferenceConfigError(f"onnx.variant phải thuộc {ONNX_VARIANTS}")
    if cfg["pytorch"].get("dtype") not in ("auto", "bf16", "fp32"):
        raise InferenceConfigError("pytorch.dtype phải là auto | bf16 | fp32 (fp16 làm T5 ra NaN)")
    for key in ("device",):
        for sec in ("pytorch", "onnx"):
            if cfg[sec].get(key, "auto") not in ("auto", "cpu", "cuda"):
                raise InferenceConfigError(f"{sec}.{key} phải là auto | cpu | cuda")
    lengths = cfg["lengths"]
    if not isinstance(lengths, dict) or not lengths:
        raise InferenceConfigError("lengths phải là dict {tên: max_new_tokens}")
    for name, value in lengths.items():
        if not isinstance(value, int) or value <= 0:
            raise InferenceConfigError(f"lengths.{name} phải là số nguyên dương (max_new_tokens)")
    if cfg["default_length"] not in lengths:
        raise InferenceConfigError("default_length phải là một khóa trong lengths")
    ld = cfg["long_document"]
    if ld["strategy"] not in STRATEGIES:
        raise InferenceConfigError(f"long_document.strategy phải thuộc {STRATEGIES}")
    if ld["chunk_length"] not in lengths:
        raise InferenceConfigError("long_document.chunk_length phải là một khóa trong lengths")
    if int(ld["chunk_max_tokens"]) > int(cfg["max_input_tokens"]):
        raise InferenceConfigError("long_document.chunk_max_tokens không được lớn hơn max_input_tokens")
    if int(ld["overlap_tokens"]) >= int(ld["chunk_max_tokens"]):
        raise InferenceConfigError("long_document.overlap_tokens phải nhỏ hơn chunk_max_tokens")
    if int(ld["max_rounds"]) < 1:
        raise InferenceConfigError("long_document.max_rounds phải ≥ 1")
    if cfg["export"]["quantization"]["isa"] not in QUANT_ISAS:
        raise InferenceConfigError(f"export.quantization.isa phải thuộc {QUANT_ISAS}")
    b = cfg["benchmark"]
    if b["length"] not in lengths:
        raise InferenceConfigError("benchmark.length phải là một khóa trong lengths")
    if b["strategy"] not in STRATEGIES:
        raise InferenceConfigError(f"benchmark.strategy phải thuộc {STRATEGIES}")
    if int(b["warmup"]) < 0 or int(b["n_documents"]) <= 0:
        raise InferenceConfigError("benchmark.warmup ≥ 0 và benchmark.n_documents > 0")


def load_inference_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_inference_config(cfg)
    return cfg


def resolve_model_path(cfg: dict[str, Any], override: str | None = None) -> str:
    """--model_path > model.pytorch_path > paths.merged_dir của train_config."""
    if override:
        return str(override)
    if cfg["model"].get("pytorch_path"):
        return str(cfg["model"]["pytorch_path"])
    with open(cfg["model"]["train_config"], encoding="utf-8") as f:
        return str(yaml.safe_load(f)["paths"]["merged_dir"])


def onnx_dirs(cfg: dict[str, Any], model_path: str) -> dict[str, Path]:
    root = cfg["model"].get("onnx_root")
    base = Path(root) if root else Path(str(Path(model_path)).rstrip("/\\") + "-onnx")
    return {v: base / v for v in ONNX_VARIANTS}


def generation_kwargs(cfg: dict[str, Any], length: str | None = None) -> dict[str, Any]:
    """Generation config chung + max_new_tokens theo mức độ dài."""
    length = length or cfg["default_length"]
    if length not in cfg["lengths"]:
        raise ValueError(f"length phải thuộc {list(cfg['lengths'])}, nhận {length!r}")
    g = cfg["generation"]
    return {
        "num_beams": int(g["num_beams"]),
        "no_repeat_ngram_size": int(g["no_repeat_ngram_size"]),
        "early_stopping": bool(g.get("early_stopping", True)),
        "max_new_tokens": int(cfg["lengths"][length]),
    }
