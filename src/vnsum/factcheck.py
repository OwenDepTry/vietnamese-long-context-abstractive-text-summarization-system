"""Kiểm tra sự thật bề mặt sau khi sinh: số liệu, ngày tháng, tên riêng trong bản tóm tắt có trong văn bản gốc không.

Chấm người trên 20 văn bản dài (results/long_v2/human_eval/) cho thấy bản tóm tắt ghép (concat) đầy đủ hơn nhưng
sai sự thật nhiều hơn; một phần lỗi là số / tên không có trong bài (sai ngày đăng ký, sai khóa đại hội, đổi tên).
Module này bắt đúng LOẠI LỖI ĐÓ, bằng đối chiếu chuỗi; nó KHÔNG bắt được lỗi quan hệ (gán nhầm lời nói, đảo ý,
trộn vai trò) vì mọi từ của câu sai kiểu đó vẫn có trong bài.

Quy tắc (ưu tiên độ chính xác: đánh dấu nhầm thì có thể làm mất câu đúng):
* Số / ngày / giờ: phải xuất hiện trong văn bản gốc (chấp nhận khác dấu phân cách hàng nghìn: 1.500 = 1500;
  3,5 vẫn là 3,5).
* Từ viết hoa toàn bộ (viết tắt, số La Mã: "VIII"): phải có trong văn bản gốc, phân biệt hoa thường.
* Chuỗi từ viết hoa liên tiếp (tên riêng): bỏ từ đầu câu (tiếng Việt viết hoa đầu câu), rồi phải tách được thành
  các cụm có trong văn bản gốc; cụm một từ phải xuất hiện VIẾT HOA trong văn bản gốc ("Mỹ Donald Trump" = "Mỹ" +
  "Donald Trump" là hợp lệ; "Nguyễn Văn Phượng" khi bài chỉ có "Trần Đức Phượng" thì không).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

_NUMBER_RE = re.compile(r"\d+(?:[.,/:\-]\d+)*")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+")


def _canonical_number(num: str) -> str:
    """1.500 / 1,500 / 1500 -> 1500; chỉ khi đúng mẫu hàng nghìn (3,5 vẫn là 3,5)."""
    return re.sub(r"[.,]", "", num) if _THOUSANDS_RE.fullmatch(num) else num
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


@dataclass
class SourceIndex:
    """Văn bản gốc đã chuẩn bị để tra nhanh."""

    text: str
    words_lower: str = field(init=False)
    capitalized: frozenset[str] = field(init=False)

    numbers: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.text = _nfc(self.text)
        self.numbers = frozenset(_canonical_number(n) for n in _NUMBER_RE.findall(self.text))
        words = _WORD_RE.findall(self.text)
        self.words_lower = " " + " ".join(w.lower() for w in words) + " "
        self.capitalized = frozenset(w for w in words if w[:1].isupper())

    def has_number(self, num: str) -> bool:
        if _canonical_number(num) in self.numbers:
            return True
        # Số nằm trong một cụm dài hơn của văn bản gốc (vd. "2018" trong "2018-2023").
        return re.search(r"(?<!\d)" + re.escape(num) + r"(?!\d)", self.text) is not None

    def has_exact_word(self, word: str) -> bool:
        return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", self.text) is not None

    def has_phrase(self, tokens: Sequence[str]) -> bool:
        return (" " + " ".join(t.lower() for t in tokens) + " ") in self.words_lower


def _segmentable(tokens: list[str], src: SourceIndex) -> bool:
    """Chuỗi tên viết hoa tách được thành các cụm có trong bài (cụm một từ phải viết hoa trong bài)."""
    n = len(tokens)
    ok = [True] + [False] * n
    for end in range(1, n + 1):
        for start in range(end):
            if not ok[start]:
                continue
            seg = tokens[start:end]
            if (len(seg) == 1 and seg[0] in src.capitalized) or (len(seg) > 1 and src.has_phrase(seg)):
                ok[end] = True
                break
    return ok[n]


def unsupported_items(sentence: str, src: SourceIndex) -> list[str]:
    """Các số / tên trong câu không tìm thấy trong văn bản gốc (rỗng = không phát hiện vấn đề)."""
    s = _nfc(sentence)
    bad = [n for n in dict.fromkeys(_NUMBER_RE.findall(s)) if not src.has_number(n)]
    words = _WORD_RE.findall(s)
    runs: list[list[str]] = []
    cur: list[str] = []
    for k, w in enumerate(words):
        if len(w) == 1:
            # Chữ cái đơn ("vitamin B", "phương án A") không phải tên riêng; cắt chuỗi tên tại đây, vì từ viết hoa
            # ngay sau nó thường là đầu câu mà bộ tách câu bỏ sót ("...vitamin B. Giảm lượng...").
            if cur:
                runs.append(cur)
                cur = []
            continue
        if w[:1].isupper() and not w.isdigit():
            prev_single = k > 0 and len(words[k - 1]) == 1
            if (k == 0 or prev_single) and not (w.isupper() and len(w) > 1):
                continue                                   # từ đầu câu: viết hoa vì là đầu câu
            cur.append(w)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        acronyms = [w for w in run if w.isupper() and len(w) > 1]
        bad += [w for w in acronyms if not src.has_exact_word(w)]
        names = [w for w in run if w not in acronyms]
        if names and not _segmentable(names, src):
            bad.append(" ".join(names))
    return list(dict.fromkeys(bad))


@dataclass
class FactCheck:
    sentences: list[str]
    unsupported: list[list[str]]

    @property
    def flagged(self) -> int:
        return sum(1 for u in self.unsupported if u)

    def kept(self) -> list[str]:
        return [s for s, u in zip(self.sentences, self.unsupported) if not u]


def check_summary(summary: str, source: str | SourceIndex, splitter: Callable[[str], list[str]]) -> FactCheck:
    src = source if isinstance(source, SourceIndex) else SourceIndex(source)
    sents = [s for s in splitter(_nfc(summary)) if s.strip()]
    return FactCheck(sents, [unsupported_items(s, src) for s in sents])


def drop_unsupported(sentences: Sequence[str], source: str | SourceIndex) -> list[str]:
    """Bỏ câu có số / tên không có căn cứ; nếu bỏ hết thì giữ nguyên (không trả bản tóm tắt rỗng)."""
    src = source if isinstance(source, SourceIndex) else SourceIndex(source)
    kept = [s for s in sentences if not unsupported_items(s, src)]
    return kept or list(sentences)
