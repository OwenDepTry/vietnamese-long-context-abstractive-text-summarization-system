"""Đọc và kiểm tra ``configs/eval.yaml`` (không import torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SYSTEM_TYPES = ("lead", "seq2seq", "llm", "pipeline")
PROVIDERS = ("anthropic", "openai_compatible")
_REQUIRED = ("seed", "paths", "eval_set", "generation", "systems", "llm", "judge", "bertscore", "aux_metrics", "qualitative")


class EvalConfigError(ValueError):
    pass


def validate_eval_config(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise EvalConfigError(f"Thiếu khóa config: {missing}")
    if cfg["generation"].get("dtype") == "fp16":
        raise EvalConfigError("generation.dtype=fp16 không được hỗ trợ: T5 dễ overflow ra NaN. Dùng auto | bf16 | fp32.")
    if cfg["generation"].get("dtype") not in ("auto", "bf16", "fp32"):
        raise EvalConfigError("generation.dtype phải là auto | bf16 | fp32")
    for name, sys_cfg in cfg["systems"].items():
        if sys_cfg.get("type") not in SYSTEM_TYPES:
            raise EvalConfigError(f"systems.{name}.type phải thuộc {SYSTEM_TYPES}")
        if sys_cfg.get("long_strategy") not in (None, "extractive", "hierarchical"):
            raise EvalConfigError(f"systems.{name}.long_strategy phải là extractive | hierarchical")
        if sys_cfg.get("long_strategy") == "hierarchical":
            need = ("chunk_max_tokens", "overlap_tokens", "chunk_max_new_tokens", "max_rounds")
            miss = [k for k in need if k not in (sys_cfg.get("hierarchical") or {})]
            if sys_cfg["type"] != "seq2seq" or miss:
                raise EvalConfigError(f"systems.{name}: long_strategy hierarchical cần type seq2seq và hierarchical.{need} (thiếu {miss})")
            if (sys_cfg.get("hierarchical") or {}).get("reduce", "generate") not in ("generate", "concat"):
                raise EvalConfigError(f"systems.{name}.hierarchical.reduce phải là generate | concat")
        if sys_cfg["type"] == "pipeline" and not sys_cfg.get("inference_config"):
            raise EvalConfigError(f"systems.{name}: type pipeline cần inference_config (vd. configs/inference.yaml)")
        if sys_cfg["type"] == "llm" and sys_cfg.get("client") not in cfg["llm"]:
            raise EvalConfigError(f"systems.{name}.client={sys_cfg.get('client')!r} không có trong mục llm")
    for name, client in cfg["llm"].items():
        if client.get("provider") not in PROVIDERS:
            raise EvalConfigError(f"llm.{name}.provider phải thuộc {PROVIDERS}")
        if not client.get("model") or not client.get("api_key_env"):
            raise EvalConfigError(f"llm.{name} cần model và api_key_env (khóa API đọc từ biến môi trường)")
        if client["provider"] == "openai_compatible" and not client.get("base_url"):
            raise EvalConfigError(f"llm.{name}: provider openai_compatible cần base_url")
    if "judge" not in cfg["llm"]:
        raise EvalConfigError("Thiếu llm.judge")
    unknown = [s for s in cfg["judge"].get("systems", []) if s not in cfg["systems"]]
    if unknown:
        raise EvalConfigError(f"judge.systems có hệ thống không tồn tại: {unknown}")
    bs = cfg["bertscore"]
    if bs.get("enabled"):
        if not isinstance(bs.get("num_layers"), int) or bs["num_layers"] <= 0:
            raise EvalConfigError("bertscore.num_layers phải là số nguyên dương, ghi rõ trong config")
        if int(bs.get("max_length", 0)) > 256:
            raise EvalConfigError("bertscore.max_length tối đa 256 với PhoBERT (max_position_embeddings=258)")
    if cfg["qualitative"]["system"] not in cfg["systems"]:
        raise EvalConfigError("qualitative.system không tồn tại trong systems")


def load_eval_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise EvalConfigError(f"{path} không phải mapping YAML")
    validate_eval_config(cfg)
    return cfg
