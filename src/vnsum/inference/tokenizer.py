"""Nạp tokenizer ViT5 trong venv phase 4 (transformers 4.57) từ thư mục do transformers 5 lưu.

Model phase 2 được lưu bằng transformers 5: ``tokenizer_config.json`` có ``extra_special_tokens`` dạng
list và ``tokenizer_class: TokenizersBackend`` — transformers 4.57 không đọc được
(``AttributeError: 'list' object has no attribute 'keys'``). ``tokenizer.json`` (mô hình Unigram +
post-processor thêm ``</s>``) thì giống hệt, nên khi cách nạp thường lỗi, ta dựng
``PreTrainedTokenizerFast`` thẳng từ ``tokenizer.json`` với các special token, rồi kiểm tra id của
chúng khớp ``config.json`` của model. Không sửa code phase 1 (``vnsum.data.tokenization``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SPECIAL_DEFAULTS = {"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}


def _special_tokens(path: Path) -> dict[str, str]:
    special = dict(_SPECIAL_DEFAULTS)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        f = path / name
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for key in special:
            v = data.get(key)
            if isinstance(v, str):
                special[key] = v
            elif isinstance(v, dict) and isinstance(v.get("content"), str):
                special[key] = v["content"]
    return special


def check_special_ids(tok: Any, path: Path) -> None:
    """pad/eos của tokenizer phải khớp config.json của model (nếu có)."""
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        return
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    for attr in ("pad_token_id", "eos_token_id"):
        want = cfg.get(attr)
        if isinstance(want, int) and getattr(tok, attr) != want:
            raise ValueError(f"Tokenizer ở {path}: {attr}={getattr(tok, attr)} khác config.json ({want})")


def load_tokenizer_compat(path: str | Path) -> Any:
    from vnsum.data.tokenization import load_tokenizer

    path = Path(path)
    try:
        tok = load_tokenizer(str(path))
    except (AttributeError, TypeError, ValueError, KeyError) as err:
        if not (path / "tokenizer.json").exists():
            raise
        logger.info("Nạp tokenizer chuẩn lỗi (%s: %s); dựng lại từ tokenizer.json", type(err).__name__, err)
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), **_special_tokens(path))
    check_special_ids(tok, path)
    return tok
