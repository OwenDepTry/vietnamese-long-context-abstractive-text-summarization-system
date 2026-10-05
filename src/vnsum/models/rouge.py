"""ROUGE cho tiếng Việt.

Tokenizer mặc định của ``rouge_score`` chỉ giữ ký tự ``[a-z0-9]``, nên làm mất
toàn bộ chữ có dấu ("Hà Nội" -> "h", "n i"). Vì vậy ở đây dùng tokenizer riêng
theo âm tiết: chuẩn NFC, viết thường, tách theo ``\\w+`` (Unicode).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
_TOKEN_RE = re.compile(r"\w+")


class VietnameseRougeTokenizer:
    """Giao diện ``tokenize(text) -> list[str]`` mà ``rouge_score`` yêu cầu."""

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(unicodedata.normalize("NFC", text or "").lower())


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """F1 trung bình (thang 0–100) của rouge1/rouge2/rougeL."""
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError("predictions và references phải cùng độ dài")
    if not predictions:
        return {k: 0.0 for k in ROUGE_TYPES}
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=False, tokenizer=VietnameseRougeTokenizer())
    totals = {k: 0.0 for k in ROUGE_TYPES}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in ROUGE_TYPES:
            totals[k] += scores[k].fmeasure
    return {k: round(100.0 * v / len(predictions), 2) for k, v in totals.items()}
