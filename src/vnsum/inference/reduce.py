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


# Số, ngày, giờ, tỷ lệ: "100", "1.500", "22/5", "10:30", "3,5".
_NUMBER_RE = re.compile(r"\d+(?:[.,/:\-]\d+)*")
_NEGATIONS = frozenset({"không", "chưa", "chẳng", "chả", "đừng", "chớ"})
# Từ ghép có "không" mà không mang nghĩa phủ định. Nhận nhầm chỉ khiến hai câu được GIỮ cả hai (an toàn).
_NOT_NEGATION = frozenset({
    ("hàng", "không"), ("phòng", "không"), ("không", "khí"), ("không", "gian"), ("không", "phận"),
    ("không", "quân"), ("không", "trung"), ("không", "lưu"), ("không", "chiến"), ("không", "kích"),
    ("không", "vận"), ("không", "ảnh"),
})


def _facts(sentence: str) -> tuple[tuple[str, ...], frozenset[str], frozenset[str]]:
    """(dãy số/ngày theo thứ tự, từ viết hoa, từ phủ định) của một câu — những thứ mà hai câu giống chữ vẫn có
    thể khác nhau."""
    text = unicodedata.normalize("NFC", sentence)
    numbers = tuple(_NUMBER_RE.findall(text))
    words = _WORD_RE.findall(text)
    # Mọi âm tiết viết hoa, KỂ CẢ đầu câu: tên riêng hay đứng đầu câu ("Nam đã nhận..." / "Lan đã nhận...").
    # So sánh không phân biệt hoa thường ở phía câu được giữ, nên "Năm 2023 ..." vẫn khớp "... năm 2023 ...".
    names = frozenset(w.lower() for w in words if w[:1].isupper() and not w.isdigit())
    low = [w.lower() for w in words]
    neg = set()
    for k, w in enumerate(low):
        if w not in _NEGATIONS:
            continue
        prev_pair = (low[k - 1], w) if k else None
        next_pair = (w, low[k + 1]) if k + 1 < len(low) else None
        if prev_pair not in _NOT_NEGATION and next_pair not in _NOT_NEGATION:
            neg.add(w)
    return numbers, names, frozenset(neg)


def _is_subsequence(small: Sequence[str], big: Sequence[str]) -> bool:
    it = iter(big)
    return all(x in it for x in small)


def facts_compatible(dropped: str, kept: str) -> bool:
    """Có bỏ được ``dropped`` mà giữ ``kept`` không (bảo thủ: không chắc thì trả False để giữ cả hai câu).

    * Số / ngày của câu bị bỏ phải xuất hiện trong câu được giữ **theo cùng thứ tự** (dãy con): "2023 đạt 100,
      2024 đạt 200" và "2023 đạt 200, 2024 đạt 100" có cùng tập số nhưng khác quan hệ năm–số liệu.
    * Mọi từ viết hoa của câu bị bỏ (kể cả từ đầu câu) phải có trong câu được giữ.
    * Hai câu có cùng từ phủ định ("đã xác nhận" ≠ "chưa xác nhận").

    Kiểm tra theo thứ tự chỉ là xấp xỉ quan hệ giữa các thành phần; không phân tích được cú pháp.
    """
    num_d, name_d, neg_d = _facts(dropped)
    num_k, _, neg_k = _facts(kept)
    kept_words = {w.lower() for w in _WORD_RE.findall(unicodedata.normalize("NFC", kept))}
    return _is_subsequence(num_d, num_k) and name_d <= kept_words and neg_d == neg_k


def _same_text(a: str, b: str) -> bool:
    return _syllables(a) == _syllables(b)


def dedupe_sentences(sentences: Sequence[str], threshold: float = 0.6, n: int = 2) -> list[str]:
    """Bỏ câu trùng / gần trùng, giữ thứ tự. Khi hai câu trùng, giữ câu DÀI hơn (nhiều thông tin hơn)
    ở vị trí của câu xuất hiện trước.

    Gần trùng = tỷ lệ bigram ≥ ``threshold`` VÀ câu bị bỏ không mang thông tin mà câu được giữ thiếu
    (``facts_compatible``). Giống chữ chưa chắc giống thông tin: "doanh thu năm 2023 đạt 100 tỷ" và
    "... năm 2024 đạt 200 tỷ" giống 80% bigram nhưng là hai sự kiện. Khi không chắc thì giữ cả hai; bước
    chọn câu phía sau vẫn cắt theo ngân sách.
    """
    kept: list[str] = []
    for s in sentences:
        s = s.strip()
        if not _syllables(s):
            continue

        def _dup(t: str) -> bool:
            if _same_text(s, t):
                return True
            if overlap_ratio(s, t, n) < threshold:
                return False
            shorter, longer = (s, t) if len(_syllables(s)) <= len(_syllables(t)) else (t, s)
            return facts_compatible(shorter, longer)

        dup = next((k for k, t in enumerate(kept) if _dup(t)), None)
        if dup is None:
            kept.append(s)
        elif len(_syllables(s)) > len(_syllables(kept[dup])):
            kept[dup] = s
    return kept


def factcheck_mode(value) -> str:
    """Chuẩn hóa tùy chọn factcheck: YAML đọc ``off`` không ngoặc thành False."""
    if value is None or value is False:
        return "off"
    return str(value).strip().lower()


def reduce_concat(
    parts: Sequence[str],
    *,
    splitter: Callable[[str], list[str]],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    select: Callable[[str, int], str],
    dedupe_threshold: float = 0.6,
    source: str | None = None,
    factcheck: str = "off",
) -> str:
    """Ghép ``parts`` (bản tóm tắt từng chunk, theo thứ tự) -> bản tóm tắt cuối ≤ ``max_tokens`` token.

    ``select(text, max_tokens)`` chọn câu khi bản ghép (đã bỏ câu trùng) vẫn quá dài — extractive
    filter, giữ thứ tự gốc. Câu cùng chủ đề với nhiều chunk khác có điểm centrality cao hơn.

    ``factcheck="drop"`` (cần ``source``): bỏ câu có số / ngày / tên riêng không có trong văn bản gốc
    (``vnsum.factcheck``) TRƯỚC khi bỏ trùng và chọn câu; nếu bỏ hết thì giữ nguyên.
    """
    factcheck = factcheck_mode(factcheck)
    if factcheck not in ("off", "drop"):
        raise ValueError(f"factcheck phải là off | drop, nhận {factcheck!r}")
    sentences = [s for p in parts if p and p.strip() for s in splitter(p.strip())]
    if factcheck == "drop" and source:
        from vnsum.factcheck import drop_unsupported

        sentences = drop_unsupported(sentences, source)
    unique = dedupe_sentences(sentences, threshold=dedupe_threshold)
    text = " ".join(unique)
    if count_tokens(text) <= int(max_tokens):
        return text
    picked = select(text, int(max_tokens))
    return " ".join(dedupe_sentences(splitter(picked), threshold=dedupe_threshold))
