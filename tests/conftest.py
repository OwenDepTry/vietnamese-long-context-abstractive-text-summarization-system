"""Fixture dùng chung. Test chạy offline bằng tokenizer giả (mỗi từ = 1 token).

Test với tokenizer ViT5 thật nằm ở ``test_vit5_tokenizer.py`` và tự skip khi
không tải được tokenizer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.data.tokenization import TokenCounter  # noqa: E402


class WhitespaceTokenizer:
    """Giả lập giao diện ``__call__`` của HF fast tokenizer: một từ = một token."""

    _TOKEN_RE = re.compile(r"\S+")

    def _encode(self, text: str, return_offsets_mapping: bool) -> dict:
        spans = [(m.start(), m.end()) for m in self._TOKEN_RE.finditer(text)]
        enc = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            enc["offset_mapping"] = spans
        return enc

    def __call__(self, text, add_special_tokens: bool = True, return_offsets_mapping: bool = False):
        if isinstance(text, (list, tuple)):
            encs = [self._encode(t, return_offsets_mapping) for t in text]
            out = {"input_ids": [e["input_ids"] for e in encs]}
            if return_offsets_mapping:
                out["offset_mapping"] = [e["offset_mapping"] for e in encs]
            return out
        return self._encode(text, return_offsets_mapping)


@pytest.fixture
def counter() -> TokenCounter:
    return TokenCounter(WhitespaceTokenizer(), reserve_special_tokens=1)


def make_document(n_sentences: int, words_per_sentence: int = 12) -> tuple[str, list[str]]:
    """Văn bản tổng hợp, mỗi câu có mã ``Sxx`` để kiểm tra thứ tự và độ phủ."""
    vocab = ["kinh tế", "giáo dục", "y tế", "thời tiết", "giao thông", "nông nghiệp", "du lịch"]
    sentences = []
    for i in range(n_sentences):
        topic = vocab[i % len(vocab)]
        filler = " ".join(f"từ{(i * 7 + j) % 23}" for j in range(max(0, words_per_sentence - 4)))
        sentences.append(f"Câu S{i:02d} về {topic} {filler}.")
    return " ".join(sentences), sentences


@pytest.fixture
def make_doc():
    return make_document
