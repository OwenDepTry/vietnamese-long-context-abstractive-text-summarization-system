import unicodedata

import pytest

from vnsum.config import load_config
from vnsum.data.preprocess import (
    collapse_whitespace,
    deduplicate,
    desegment,
    filter_records,
    normalize_text,
    normalize_tone_style,
    remove_control_chars,
    split_sentences,
    strip_html,
)

PRE_CFG = {
    "unicode_form": "NFC",
    "tone_style": "old",
    "strip_html": True,
    "remove_control_chars": True,
    "collapse_whitespace": True,
    "keep_newlines": True,
}


# ------------------------------- normalize -------------------------------- #


def test_nfc_composes_decomposed_text():
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    assert decomposed != "Tiếng Việt"
    out = normalize_text(decomposed, PRE_CFG)
    assert out == "Tiếng Việt"
    assert unicodedata.is_normalized("NFC", out)


@pytest.mark.parametrize(
    "new, old",
    [("hoà", "hòa"), ("khoẻ", "khỏe"), ("thuỷ", "thủy"), ("HOÁ", "HÓA"), ("Tuỳ", "Tùy"), ("loè", "lòe")],
)
def test_tone_style_roundtrip(new, old):
    assert normalize_tone_style(new, "old") == old
    assert normalize_tone_style(old, "new") == new
    assert normalize_tone_style(old, "old") == old  # idempotent


@pytest.mark.parametrize("word", ["hoàng", "khuyến", "quý", "quả", "thuế", "nguyễn", "toán"])
def test_tone_style_leaves_unambiguous_words(word):
    assert normalize_tone_style(word, "old") == word
    assert normalize_tone_style(word, "new") == word


def test_tone_style_none_is_noop():
    assert normalize_tone_style("hoà bình", None) == "hoà bình"


def test_strip_html_keeps_text_and_comparisons():
    raw = "<p>Giá &amp; lãi</p><script>alert(1)</script><br/>x < 5 và y > 3"
    out = collapse_whitespace(strip_html(raw))
    assert "<" not in out.replace("x < 5", "")
    assert "Giá & lãi" in out
    assert "alert" not in out
    assert "x < 5 và y > 3" in out


def test_remove_control_chars():
    raw = "a​b﻿c\x00d\te\r\nf"
    assert remove_control_chars(raw) == "abcd e\n\nf"


def test_collapse_whitespace_keeps_single_newlines():
    assert collapse_whitespace("  a   b \n\n\n  c  ") == "a b\nc"
    assert collapse_whitespace("a \n b", keep_newlines=False) == "a b"


def test_desegment_vietnews_style():
    raw = 'Thương đã đột_nhập vào nhà , lấy hơn 8 triệu đồng ( tiền_mặt ) . Ông nói " không biết " .'
    assert desegment(raw) == 'Thương đã đột nhập vào nhà, lấy hơn 8 triệu đồng (tiền mặt). Ông nói "không biết".'


def test_desegment_abbreviation_joined_with_underscore():
    # Gặp thật trong nam194/vietnews (train): công cụ tách từ nối "TP." với tên riêng.
    assert desegment("trú tại tổ 14 , TP._Pleiku , tỉnh Gia_Lai") == "trú tại tổ 14, TP. Pleiku, tỉnh Gia Lai"


def test_normalize_full_pipeline():
    raw = "<div>Hoà  bình​</div>\n\n<p>Sức khoẻ</p>"
    assert normalize_text(raw, PRE_CFG) == "Hòa bình\nSức khỏe"


def test_normalize_empty():
    assert normalize_text(None, PRE_CFG) == ""
    assert normalize_text("   ", PRE_CFG) == ""


# ------------------------------ tách câu ---------------------------------- #


def test_split_basic():
    text = "Hôm nay trời đẹp. Tôi đi học! Bạn có đi không? Có."
    assert split_sentences(text) == ["Hôm nay trời đẹp.", "Tôi đi học!", "Bạn có đi không?", "Có."]


def test_split_respects_abbreviations_and_numbers():
    text = "PGS.TS. Nguyễn Văn A làm việc tại TP. Hồ Chí Minh. Giá là 1.000 đồng. Ông Trần V. B đồng ý."
    assert split_sentences(text) == [
        "PGS.TS. Nguyễn Văn A làm việc tại TP. Hồ Chí Minh.",
        "Giá là 1.000 đồng.",
        "Ông Trần V. B đồng ý.",
    ]


def test_split_on_newlines_and_quotes():
    text = 'Ông nói: "Tôi đồng ý." Sau đó ông rời đi\nĐoạn mới bắt đầu.'
    assert split_sentences(text) == ['Ông nói: "Tôi đồng ý."', "Sau đó ông rời đi", "Đoạn mới bắt đầu."]


def test_split_does_not_break_before_lowercase():
    assert split_sentences("Kết quả là 3.5 điểm. và chưa xong.") == ["Kết quả là 3.5 điểm. và chưa xong."]


def test_split_empty():
    assert split_sentences("") == []


# ------------------------------ lọc mẫu ----------------------------------- #

FILTER_CFG = {
    "length_unit": "words",
    "min_document_words": 3,
    "min_summary_words": 1,
    "drop_summary_longer_than_document": True,
}


def test_filter_records_reasons():
    recs = [
        {"id": "ok", "document": "một hai ba bốn năm", "summary": "một hai"},
        {"id": "empty_doc", "document": "  ", "summary": "x"},
        {"id": "empty_sum", "document": "một hai ba bốn", "summary": ""},
        {"id": "long_sum", "document": "một hai ba", "summary": "một hai ba bốn năm"},
        {"id": "short_doc", "document": "một hai", "summary": "một"},
    ]
    kept, reasons = filter_records(recs, FILTER_CFG)
    assert [r["id"] for r in kept] == ["ok"]
    assert reasons == {
        "empty_document": 1,
        "empty_summary": 1,
        "summary_longer_than_document": 1,
        "short_document": 1,
    }


def test_deduplicate_within_and_across_splits():
    splits = {
        "train": [
            {"id": "t1", "document": "Văn bản A", "summary": "a"},
            {"id": "t2", "document": "văn  bản a", "summary": "a"},  # trùng t1 (khác hoa/khoảng trắng)
            {"id": "t3", "document": "Văn bản B", "summary": "b"},  # trùng test
        ],
        "test": [{"id": "s1", "document": "Văn bản B", "summary": "b"}],
    }
    cfg = {"enabled": True, "key": "document", "cross_split": True, "split_priority": ["test", "validation", "train"]}
    out, dropped = deduplicate(splits, cfg)
    assert [r["id"] for r in out["train"]] == ["t1"]
    assert [r["id"] for r in out["test"]] == ["s1"]
    assert dropped == {"duplicate_within_train": 1, "duplicate_cross_split_train": 1}
    assert list(out) == ["train", "test"]  # giữ thứ tự split ban đầu


def test_repo_config_is_valid():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "data.yaml")
    assert cfg["preprocess"]["word_segmentation"]["enabled"] is False  # không tách từ mặc định cho ViT5
    assert cfg["tokenizer"]["name_or_path"] == "VietAI/vit5-base"
