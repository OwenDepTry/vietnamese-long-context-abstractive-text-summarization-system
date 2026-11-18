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


# ViT5 hay tách "48.000" thành "48. 000" khi giải mã: ghép lại trước khi đối chiếu số.
# Chỉ dấu CHẤM (dấu hàng nghìn tiếng Việt); "130, 122, 105" là danh sách số, không được ghép. Lookbehind không
# tiêu thụ chữ số nên chuỗi "1. 000. 000" cũng ghép hết.
_ARTIFACT_RE = re.compile(r"(?<=\d)\. (?=\d{3}(?!\d))")
_DECIMAL_RE = re.compile(r"\d+[.,]\d+")
_DATE_RE = re.compile(r"(\d{1,2}/\d{1,2})/(\d{4})")       # 7/1/2015 -> ngày "7/1", năm "2015"
_MONTH_YEAR_RE = re.compile(r"\d{1,2}/(\d{4})")            # 1/2015 -> năm "2015"


_DAY_MONTH_RE = re.compile(r"\d{1,2}-\d{1,2}(?:-\d{4})?")


def _canonical_number(num: str) -> str:
    """1.500 / 1,500 / 1500 -> 1500 (đúng mẫu hàng nghìn); 3,5 / 3.5 -> 3.5 (thập phân, không phải 35);
    ngày viết bằng gạch nối -> gạch chéo (17-11 = 17/11, 1-1-2019 = 1/1/2019)."""
    if _DAY_MONTH_RE.fullmatch(num):
        return num.replace("-", "/")
    if _THOUSANDS_RE.fullmatch(num):
        return re.sub(r"[.,]", "", num)
    if _DECIMAL_RE.fullmatch(num):
        return num.replace(",", ".")
    return num


def _parts_of(token: str) -> set[str]:
    """Các số được coi là CÓ trong một cụm số của văn bản gốc, ngoài chính cụm đó: từng đầu của khoảng
    ("2018-2023" -> 2018, 2023) và ngày / năm của ngày đầy đủ ("7/1/2015" -> 7/1, 2015). KHÔNG gồm phần của số
    thập phân hay hàng nghìn: "5" không có trong "3,5", "500" không có trong "1.500"."""
    out: set[str] = set()
    pieces = token.split("-")
    if _DAY_MONTH_RE.fullmatch(token):           # "17-11-2018" vừa là ngày (17/11/2018) vừa có thể là khoảng
        pieces.append(_canonical_number(token))
    for piece in pieces:
        out.add(_canonical_number(piece))
        m = _DATE_RE.fullmatch(piece)
        if m:
            out.update(m.groups())
        m = _MONTH_YEAR_RE.fullmatch(piece)
        if m:
            out.add(m.group(1))
    return out
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
    number_parts: frozenset[str] = field(init=False)
    lower_words: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.text = _ARTIFACT_RE.sub(".", _nfc(self.text))   # văn bản gốc cũng có "1/2. 000"
        raw = _NUMBER_RE.findall(self.text)
        self.numbers = frozenset(_canonical_number(n) for n in raw)
        self.number_parts = frozenset(p for n in raw for p in _parts_of(n))
        words = _WORD_RE.findall(self.text)
        self.words_lower = " " + " ".join(w.lower() for w in words) + " "
        self.capitalized = frozenset(w for w in words if w[:1].isupper())
        self.lower_words = frozenset(w.lower() for w in words)

    def has_number(self, num: str) -> bool:
        c = _canonical_number(num)
        return c in self.numbers or c in self.number_parts

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


# Họ và tên đệm / tên phổ biến của người Việt. Từ viết hoa ĐẦU CÂU chỉ được kiểm tra khi thuộc danh sách này: viết
# hoa đầu câu thì không phân biệt được tên với động từ ("Nhận diện...", "Cân nhắc..." trong WikiLingua) bằng đối
# chiếu chuỗi, nên chỉ bắt trường hợp rất có khả năng là tên người ("Lan đã nhận học bổng" khi bài nói về "Nam").
VIETNAMESE_NAMES = frozenset("""
nguyễn trần lê phạm hoàng huỳnh phan vũ võ đặng bùi đỗ hồ ngô dương lý đinh trịnh đoàn lâm mai lương tạ tô
châu quách triệu khổng thái kiều cao hà vương
lan hùng dũng tuấn hương hằng trang linh thảo huyền nga nhung oanh phương quyên quỳnh thúy thủy tâm thu
loan liên hạnh hiền hồng hà hải hiếu hiệp hoa hòa hoà hùng huy hưng khánh khoa khôi kiên lâm long luân
mạnh minh my mỹ nam nghĩa ngọc nhân nhi nhung phong phúc phước quân quang quốc sơn tài thắng thành thanh
thảo thịnh thông thư trí trinh trung tú tùng tuyết uyên vân việt vinh vy xuân yến đạt đức duy dương giang
hạnh hậu hiển hoàng khang kim lộc lực ly mai my na ninh oanh phát phi phượng quý sang sinh tân thái thiện
thơ thuận thương tiến toàn trâm trọng tuệ tường văn vũ yên
""".split())


def _initial_is_name(word: str, src: SourceIndex) -> bool:
    """Từ viết hoa đầu câu là tên người lạ: thuộc danh sách tên phổ biến, không xuất hiện viết hoa trong bài."""
    return word.lower() in VIETNAMESE_NAMES and word not in src.capitalized


def unsupported_items(sentence: str, src: SourceIndex, *, check_initial: bool = False) -> list[str]:
    """Các số / tên trong câu không tìm thấy trong văn bản gốc (rỗng = không phát hiện vấn đề).

    ``check_initial``: kiểm tra cả từ viết hoa đầu câu khi nó là tên người phổ biến mà bài không có (bắt "Lan đã
    nhận học bổng" khi bài nói về "Nam"). False: bỏ qua mọi từ đầu câu như trước."""
    s = _ARTIFACT_RE.sub(".", _nfc(sentence))
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
                if not (check_initial and _initial_is_name(w, src)):
                    continue                               # từ đầu câu: viết hoa vì là đầu câu
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


def check_summary(summary: str, source: str | SourceIndex, splitter: Callable[[str], list[str]], *,
                  check_initial: bool = False) -> FactCheck:
    src = source if isinstance(source, SourceIndex) else SourceIndex(source)
    sents = [s for s in splitter(_ARTIFACT_RE.sub(".", _nfc(summary))) if s.strip()]
    return FactCheck(sents, [unsupported_items(s, src, check_initial=check_initial) for s in sents])


@dataclass
class DropResult:
    sentences: list[str]
    dropped: int
    all_flagged: bool   # mọi câu đều bị đánh dấu -> giữ nguyên (không trả rỗng); bản tóm tắt KHÔNG qua kiểm tra

    def __iter__(self):   # tương thích: list(drop_unsupported(...)) như trước
        return iter(self.sentences)

    def __eq__(self, other):
        return list(self.sentences) == list(other) if isinstance(other, list) else NotImplemented


def drop_unsupported(sentences: Sequence[str], source: str | SourceIndex) -> DropResult:
    """Bỏ câu có số / tên không có căn cứ. Nếu MỌI câu đều bị đánh dấu thì giữ nguyên (không trả bản tóm tắt
    rỗng) và đặt ``all_flagged`` để nơi gọi báo rõ là bản tóm tắt không qua được kiểm tra."""
    src = source if isinstance(source, SourceIndex) else SourceIndex(source)
    # Câu bị bộ tách câu cắt ở "48. 000" (lỗi giải mã của ViT5): kiểm tra và giữ / bỏ cả nhóm như một câu.
    groups: list[list[str]] = []
    for sent in sentences:
        if groups and re.search(r"\d[.,]\s*$", groups[-1][-1]) and re.match(r"\s*\d{3}(?!\d)", sent):
            groups[-1].append(sent)
        else:
            groups.append([sent])
    kept = [s for g in groups if not unsupported_items(" ".join(g), src) for s in g]
    if kept:
        return DropResult(kept, len(sentences) - len(kept), False)
    return DropResult(list(sentences), 0, bool(sentences))
