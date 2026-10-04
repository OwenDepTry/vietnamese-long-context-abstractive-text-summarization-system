"""Tiền xử lý văn bản tiếng Việt.

Gồm các bước sau:

* ``normalize_text``: bỏ HTML, chuẩn Unicode NFC, bỏ ký tự điều khiển, thống
  nhất kiểu đặt dấu thanh (cũ/mới), gộp khoảng trắng thừa. Có thể khôi phục
  văn bản đã tách từ (``desegment``).
* ``split_sentences``: tách câu theo luật (mặc định) hoặc bằng underthesea.
* ``filter_records`` / ``deduplicate``: lọc mẫu lỗi (rỗng, summary dài hơn
  document, trùng lặp).
* ``word_segment``: tách từ TÙY CHỌN cho thành phần cần nó (PhoBERT). Không áp
  dụng cho input của ViT5.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Unicode / HTML / ký tự điều khiển / khoảng trắng
# --------------------------------------------------------------------------- #

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BLOCK_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# Chỉ khớp thẻ thật (<p>, </div>, <br/>, <a href=...>); "a < b > c" không bị xóa.
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>")
_BLOCK_TAG_RE = re.compile(r"</?(?:p|div|br|li|ul|ol|tr|h[1-6]|section|article)\b[^<>]*>", re.IGNORECASE)

_INLINE_WS_RE = re.compile(r"[^\S\n]+")  # mọi whitespace trừ \n
_MULTI_NL_RE = re.compile(r"\s*\n\s*")


def strip_html(text: str) -> str:
    """Bỏ comment, script/style, thẻ HTML; giải mã entity (&amp; -> &)."""
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_BLOCK_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return html.unescape(text)


def remove_control_chars(text: str) -> str:
    """Bỏ ký tự điều khiển (Cc) và định dạng vô hình (Cf: zero-width, BOM, soft hyphen).

    Giữ ``\\n``; ``\\t``/``\\r`` và các ký tự phân cách dòng Unicode được đổi thành khoảng trắng/xuống dòng.
    """
    out: list[str] = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
        elif ch in "\t\r\x0b\x0c":
            out.append(" " if ch != "\r" else "\n")
        elif ch in "  \x85":
            out.append("\n")
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str, keep_newlines: bool = True) -> str:
    text = _INLINE_WS_RE.sub(" ", text)
    if keep_newlines:
        text = _MULTI_NL_RE.sub("\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Thống nhất kiểu đặt dấu thanh (cũ / mới)
# --------------------------------------------------------------------------- #

_TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}  # huyền, sắc, ngã, hỏi, nặng
_WORD_RE = re.compile(r"\w+")
# Vần oa/oe/uy KHÔNG có phụ âm cuối là nơi hai kiểu khác nhau:
#   cũ: hòa, khỏe, thúy     mới: hoà, khoẻ, thuý
# Khi có phụ âm cuối (hoàng, khuyến) hay sau "q" (quý) hai kiểu trùng nhau.
_OPEN_CLUSTER_RE = re.compile(r"(oa|oe|uy)$")


@lru_cache(maxsize=200_000)
def _retone_word(word: str, style: str) -> str:
    nfd = unicodedata.normalize("NFD", word)
    tones = [c for c in nfd if c in _TONE_MARKS]
    if len(tones) != 1:
        return word
    tone = tones[0]
    base = unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _TONE_MARKS))
    m = _OPEN_CLUSTER_RE.search(base.lower())
    if m is None:
        return word
    start = m.start()
    if start > 0 and base[start - 1].lower() == "q":
        return word
    pos = start if style == "old" else start + 1
    target = unicodedata.normalize("NFD", base[pos]) + tone
    return unicodedata.normalize("NFC", base[:pos] + target + base[pos + 1 :])


def normalize_tone_style(text: str, style: str | None) -> str:
    """Đưa các vần oa/oe/uy mở về cùng một kiểu đặt dấu (``old`` | ``new``); ``None`` -> giữ nguyên."""
    if style is None:
        return text
    if style not in ("old", "new"):
        raise ValueError(f"tone_style không hợp lệ: {style!r}")
    return _WORD_RE.sub(lambda m: _retone_word(m.group(0), style), text)


# --------------------------------------------------------------------------- #
# Khôi phục văn bản đã tách từ / tách dấu câu (VietNews trên Hub)
# --------------------------------------------------------------------------- #

# "Hà_Nội" và cả viết tắt có dấu chấm bị nối: "TP._Pleiku" -> "TP. Pleiku".
_SEG_UNDERSCORE_RE = re.compile(r"(?<=[\w.])_(?=\w)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%…)\]}»”’])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{«“‘])\s+")


def desegment(text: str) -> str:
    """``"Hà_Nội , ngày ( 5/10 ) ."`` -> ``"Hà Nội, ngày (5/10)."``"""
    text = _SEG_UNDERSCORE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    parts = text.split('"')
    if len(parts) > 1 and len(parts) % 2 == 1:  # số dấu " chẵn -> ghép cặp được
        text = '"'.join(p.strip() if i % 2 == 1 else p for i, p in enumerate(parts))
    return text


# --------------------------------------------------------------------------- #
# Hàm chuẩn hóa tổng
# --------------------------------------------------------------------------- #


def normalize_text(text: str | None, cfg: dict[str, Any], desegment_text: bool = False) -> str:
    """Chuẩn hóa một chuỗi theo ``cfg`` (mục ``preprocess`` trong config)."""
    if not text:
        return ""
    if cfg.get("strip_html", True):
        text = strip_html(text)
    text = unicodedata.normalize(cfg.get("unicode_form", "NFC"), text)
    if cfg.get("remove_control_chars", True):
        text = remove_control_chars(text)
    if desegment_text:
        text = desegment(text)
    text = normalize_tone_style(text, cfg.get("tone_style"))
    if cfg.get("collapse_whitespace", True):
        text = collapse_whitespace(text, keep_newlines=cfg.get("keep_newlines", True))
    return text


# --------------------------------------------------------------------------- #
# Tách câu
# --------------------------------------------------------------------------- #

# Viết tắt thường gặp, kết thúc bằng "." nhưng không phải cuối câu (so sánh chữ thường).
_ABBREVIATIONS = frozenset(
    {
        "tp", "tx", "tt", "ths", "ts", "pgs", "gs", "bs", "ks", "ls", "cn", "q", "p", "h",
        "mr", "mrs", "ms", "dr", "st", "no", "vs", "e.g", "i.e", "ubnd", "hđnd", "tw",
    }
)
_CANDIDATE_RE = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
_OPENERS = "\"'“‘([«-–— "


def _is_abbreviation(prefix: str) -> bool:
    last = prefix.rsplit(None, 1)[-1] if prefix.strip() else ""
    if not last.endswith("."):
        return False
    core = last.rstrip(".").lstrip("\"'“‘([")
    if not core:
        return False
    if len(core) == 1 and core.isupper():  # chữ cái viết tắt tên: "Nguyễn V. A"
        return True
    lowered = core.lower()
    return lowered in _ABBREVIATIONS or lowered.rsplit(".", 1)[-1] in _ABBREVIATIONS


def _split_paragraph_rule(para: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for m in _CANDIDATE_RE.finditer(para):
        rest = para[m.end() :].lstrip(_OPENERS)
        if not rest or not (rest[0].isupper() or rest[0].isdigit()):
            continue
        punct = m.group(0).strip()
        if punct.startswith(".") and not punct.startswith("..") and _is_abbreviation(para[start : m.start() + 1]):
            continue
        sentences.append(para[start : m.end()].strip())
        start = m.end()
    tail = para[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str, backend: str = "rule", min_chars: int = 2) -> list[str]:
    """Tách câu. Xuống dòng luôn được coi là ranh giới câu.

    ``rule``: luật nhẹ cho tiếng Việt (dấu kết câu + chữ hoa/số tiếp theo, bỏ qua viết tắt).
    ``underthesea``: dùng ``underthesea.sent_tokenize`` cho từng đoạn.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if backend == "rule":
        splitter = _split_paragraph_rule
    elif backend == "underthesea":
        from underthesea import sent_tokenize  # import trễ

        splitter = sent_tokenize
    else:
        raise ValueError(f"sentence_split.backend không hợp lệ: {backend!r}")
    out: list[str] = []
    for para in paragraphs:
        out.extend(s.strip() for s in splitter(para) if len(s.strip()) >= min_chars)
    return out


# --------------------------------------------------------------------------- #
# Lọc mẫu lỗi và trùng lặp
# --------------------------------------------------------------------------- #


def _length(text: str, unit: str) -> int:
    return len(text.split()) if unit == "words" else len(text)


def check_record(rec: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Trả lý do loại mẫu (``empty_document``...), hoặc ``None`` nếu mẫu hợp lệ."""
    unit = cfg.get("length_unit", "words")
    doc, summ = rec.get("document") or "", rec.get("summary") or ""
    if not doc.strip():
        return "empty_document"
    if not summ.strip():
        return "empty_summary"
    doc_len, summ_len = _length(doc, unit), _length(summ, unit)
    if unit == "words":
        if doc_len < int(cfg.get("min_document_words", 0)):
            return "short_document"
        if summ_len < int(cfg.get("min_summary_words", 0)):
            return "short_summary"
    if cfg.get("drop_summary_longer_than_document", True) and summ_len > doc_len:
        return "summary_longer_than_document"
    return None


def filter_records(records: Iterable[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    kept: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    for rec in records:
        reason = check_record(rec, cfg)
        if reason is None:
            kept.append(rec)
        else:
            reasons[reason] += 1
    return kept, reasons


def dedup_key(rec: dict[str, Any], key: str = "document") -> str:
    text = rec["document"] if key == "document" else f"{rec['document']}\x00{rec['summary']}"
    canon = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def deduplicate(
    splits: dict[str, list[dict[str, Any]]], cfg: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    """Bỏ mẫu trùng trong từng split và (tùy chọn) giữa các split.

    Với ``cross_split``, split đứng trước trong ``split_priority`` giữ mẫu, split
    sau bỏ mẫu: một bài trong test không được xuất hiện lại trong train.
    """
    if not cfg.get("enabled", True):
        return splits, Counter()
    key = cfg.get("key", "document")
    priority = [s for s in cfg.get("split_priority", []) if s in splits]
    order = priority + [s for s in splits if s not in priority]
    seen_global: set[str] = set()
    dropped: Counter = Counter()
    out: dict[str, list[dict[str, Any]]] = {}
    for split in order:
        seen_local: set[str] = set()
        kept: list[dict[str, Any]] = []
        for rec in splits[split]:
            k = dedup_key(rec, key)
            if k in seen_local:
                dropped[f"duplicate_within_{split}"] += 1
                continue
            if cfg.get("cross_split", True) and k in seen_global:
                dropped[f"duplicate_cross_split_{split}"] += 1
                continue
            seen_local.add(k)
            kept.append(rec)
        seen_global |= seen_local
        out[split] = kept
    return {s: out[s] for s in splits}, dropped


# --------------------------------------------------------------------------- #
# Tách từ (TÙY CHỌN, chỉ cho PhoBERT)
# --------------------------------------------------------------------------- #


def word_segment(text: str, backend: str = "underthesea") -> str:
    """Tách từ kiểu PhoBERT (``"Hà_Nội"``). KHÔNG dùng cho input ViT5."""
    if not text:
        return ""
    if backend == "underthesea":
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    if backend == "pyvi":
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)
    raise ValueError(f"word_segmentation.backend không hợp lệ: {backend!r}")
