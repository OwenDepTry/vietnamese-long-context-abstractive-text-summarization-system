"""Bước reduce của tóm tắt phân cấp: ghép bản tóm tắt các chunk thành bản tóm tắt cuối.

Hai cách (``long_document.reduce`` / ``hierarchical.reduce``):

* ``generate`` (cách cũ): ghép các bản tóm tắt chunk rồi đưa qua model thêm một lần.
  Đo trên 225 văn bản dài (``results/long/report.md``): bản cuối chỉ ~16 âm tiết so với ~53 của
  reference, ROUGE-L 13.48 so với 20.70 của extractive. Nguyên nhân: nén hai lần, và model chưa từng
  được train trên input dạng "ghép nhiều bản tóm tắt" nên coi nó như một bài báo và chỉ giữ câu đầu.
* ``concat``: KHÔNG sinh lại. Ghép bản tóm tắt các chunk theo thứ tự, bỏ câu trùng / gần trùng (chunk
  có overlap nên thường sinh câu giống nhau), rồi nếu vẫn dài hơn ngân sách thì chọn câu bằng
  extractive filter (BM25-centrality, giữ thứ tự gốc).

Chỉ dùng thư viện chuẩn; không phụ thuộc torch, test chạy offline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Sequence

REDUCE_MODES = ("generate", "concat")

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _syllables(sentence: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFC", sentence).lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def overlap_ratio(a: str, b: str, n: int = 2) -> float:
    """Tỷ lệ n-gram (âm tiết) của câu NGẮN hơn có mặt trong câu kia (containment, 0–1).

    Dùng containment thay vì Jaccard: chunk sau hay sinh lại câu của chunk trước nhưng thêm/bớt vài chữ,
    và câu ngắn nằm gọn trong câu dài vẫn là trùng ý.
    """
    ga, gb = _ngrams(_syllables(a), n), _ngrams(_syllables(b), n)
    if not ga or not gb:
        return 0.0
    small, big = (ga, gb) if len(ga) <= len(gb) else (gb, ga)
    return len(small & big) / len(small)


def dedupe_sentences(sentences: Sequence[str], threshold: float = 0.6, n: int = 2) -> list[str]:
    """Bỏ câu trùng / gần trùng, giữ thứ tự. Khi hai câu trùng, giữ câu DÀI hơn (nhiều thông tin hơn)
    ở vị trí của câu xuất hiện trước."""
    kept: list[str] = []
    for s in sentences:
        s = s.strip()
        if not _syllables(s):
            continue
        dup = next((k for k, t in enumerate(kept) if overlap_ratio(s, t, n) >= threshold), None)
        if dup is None:
            kept.append(s)
        elif len(_syllables(s)) > len(_syllables(kept[dup])):
            kept[dup] = s
    return kept


def reduce_concat(
    parts: Sequence[str],
    *,
    splitter: Callable[[str], list[str]],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    select: Callable[[str, int], str],
    dedupe_threshold: float = 0.6,
) -> str:
    """Ghép ``parts`` (bản tóm tắt từng chunk, theo thứ tự) -> bản tóm tắt cuối ≤ ``max_tokens`` token.

    ``select(text, max_tokens)`` chọn câu khi bản ghép (đã bỏ câu trùng) vẫn quá dài — extractive
    filter, giữ thứ tự gốc. Câu cùng chủ đề với nhiều chunk khác có điểm centrality cao hơn.
    """
    sentences = [s for p in parts if p and p.strip() for s in splitter(p.strip())]
    unique = dedupe_sentences(sentences, threshold=dedupe_threshold)
    text = " ".join(unique)
    if count_tokens(text) <= int(max_tokens):
        return text
    picked = select(text, int(max_tokens))
    return " ".join(dedupe_sentences(splitter(picked), threshold=dedupe_threshold))
