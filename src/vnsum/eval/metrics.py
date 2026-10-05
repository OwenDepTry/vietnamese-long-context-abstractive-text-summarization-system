"""ROUGE theo từng mẫu (tokenizer tiếng Việt giữ dấu) và các chỉ số phụ.

Tokenizer mặc định của ``rouge_score`` chỉ giữ ``[a-z0-9]`` nên xóa mọi ký tự có
dấu: "Hà Nội" -> "h n i", trùng với "Hè Nổi". Ở đây dùng ``VietnameseRougeTokenizer``
(chuẩn NFC, viết thường, tách theo ``\\w+`` Unicode — tức tách theo khoảng trắng và
dấu câu, giữ nguyên dấu), là cùng tokenizer đã dùng để tính ROUGE trong lúc train.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from vnsum.models.rouge import ROUGE_TYPES, VietnameseRougeTokenizer

_TOKENIZER = VietnameseRougeTokenizer()


def syllables(text: str) -> list[str]:
    """Âm tiết (đơn vị từ của tiếng Việt khi chưa tách từ), viết thường, giữ dấu."""
    return _TOKENIZER.tokenize(text)


def rouge_per_sample(predictions: Sequence[str], references: Sequence[str]) -> list[dict[str, float]]:
    """F1 (0–100) của rouge1/rouge2/rougeL cho từng cặp."""
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError("predictions và references phải cùng độ dài")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=False, tokenizer=_TOKENIZER)
    out = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        out.append({k: 100.0 * s[k].fmeasure for k in ROUGE_TYPES})
    return out


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repetition_rate(text: str, n: int = 3) -> float:
    """Tỷ lệ n-gram trong summary là bản lặp của một n-gram đã xuất hiện trước đó (0–100)."""
    grams = ngrams(syllables(text), n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values())
    return 100.0 * repeated / len(grams)


def novel_ngram_rate(summary: str, document: str, n: int) -> float:
    """Tỷ lệ n-gram (không trùng lặp) của summary KHÔNG xuất hiện trong document (0–100)."""
    grams = set(ngrams(syllables(summary), n))
    if not grams:
        return 0.0
    source = set(ngrams(syllables(document), n))
    return 100.0 * len(grams - source) / len(grams)


def aux_per_sample(summary: str, document: str, repetition_n: int, novel_ns: Sequence[int]) -> dict[str, float]:
    row = {"length_syllables": float(len(syllables(summary))), f"repetition_{repetition_n}gram": repetition_rate(summary, repetition_n)}
    for n in novel_ns:
        row[f"novel_{n}gram"] = novel_ngram_rate(summary, document, n)
    return row
