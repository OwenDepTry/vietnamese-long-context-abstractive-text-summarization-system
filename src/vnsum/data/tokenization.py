"""Đếm độ dài theo token của tokenizer ViT5 (không đếm theo từ).

``TokenCounter`` bọc một tokenizer kiểu Hugging Face (callable, hỗ trợ
``add_special_tokens`` và ``return_offsets_mapping``). Nhờ vậy unit test có thể
inject một tokenizer giả, chạy offline, trong khi pipeline thật dùng
``VietAI/vit5-base``.
"""

from __future__ import annotations

import os
from typing import Any, Sequence


class TokenCounter:
    """Đếm và cắt văn bản theo token, không tính special token."""

    def __init__(self, tokenizer: Any, reserve_special_tokens: int = 1) -> None:
        self.tokenizer = tokenizer
        self.reserve_special_tokens = int(reserve_special_tokens)

    def budget(self, max_tokens: int) -> int:
        """Số token văn bản tối đa khi model cần ``max_tokens`` tính cả special token."""
        return max(0, int(max_tokens) - self.reserve_special_tokens)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_many(self, texts: Sequence[str], batch_size: int = 256) -> list[int]:
        out: list[int] = []
        for start in range(0, len(texts), batch_size):
            batch = [t or "" for t in texts[start : start + batch_size]]
            enc = self.tokenizer(batch, add_special_tokens=False)["input_ids"]
            out.extend(len(ids) for ids in enc)
        return out

    def truncate(self, text: str, max_tokens: int) -> str:
        """Cắt ``text`` còn tối đa ``max_tokens`` token, cắt tại offset ký tự của token."""
        if max_tokens <= 0 or not text:
            return ""
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        if len(offsets) <= max_tokens:
            return text
        end = offsets[max_tokens - 1][1]
        cut = text[:end].rstrip()
        # Phòng trường hợp ghép token tại ranh giới làm lệch số đếm.
        while cut and self.count(cut) > max_tokens:
            cut = cut[:-1].rstrip()
        return cut


def load_tokenizer(name_or_path: str, token_env: str | None = None, cache_dir: str | None = None) -> Any:
    """Tải tokenizer từ Hub/local. Token HF đọc từ biến môi trường, không bao giờ từ code.

    Nạp thẳng ``tokenizer.json`` bằng ``PreTrainedTokenizerFast`` thay vì
    ``AutoTokenizer``. Ở transformers v5, ``AutoTokenizer`` chuyển sang lớp
    ``T5Tokenizer``, mà lớp này dựng cứng mô hình Unigram nên gãy với
    ``tokenizer.json`` của VietAI/vit5-base (``TypeError: 'dict' object is not an
    instance of 'Sequence'``). Nạp file trực tiếp giữ nguyên mô hình và
    post-processor (thêm ``</s>``) như trên Hub, chạy được cả v4 lẫn v5.
    """
    from transformers import PreTrainedTokenizerFast  # import trễ: unit test không cần transformers

    token = os.environ.get(token_env) if token_env else None
    tok = PreTrainedTokenizerFast.from_pretrained(name_or_path, token=token or None, cache_dir=cache_dir)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer {name_or_path} không phải fast tokenizer; cần offset_mapping để cắt chính xác."
        )
    return tok


def build_token_counter(cfg: dict[str, Any]) -> TokenCounter:
    tok_cfg = cfg["tokenizer"]
    tok = load_tokenizer(tok_cfg["name_or_path"], cfg["hf"].get("token_env"))
    return TokenCounter(tok, reserve_special_tokens=tok_cfg["reserve_special_tokens"])
