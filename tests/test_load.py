"""Kiểm tra phần chuẩn hóa schema của loader (không cần mạng)."""

from collections import Counter

from vnsum.data.load import assign_split, standardize_vietnews, standardize_wikilingua

RATIOS = {"train": 0.9, "validation": 0.05, "test": 0.05}


def test_vietnews_schema():
    row = {"guid": 7, "title": "t", "abstract": "tóm tắt", "article": "nội dung"}
    rec = standardize_vietnews(row, "train", {"id": "guid", "document": "article", "summary": "abstract"}, "vietnews")
    assert rec == {"id": "vietnews-train-7", "document": "nội dung", "summary": "tóm tắt"}


def test_wikilingua_sections_both_layouts():
    as_list = {"url": "u1", "article": [{"document": "d0", "summary": "s0"}, {"document": "d1", "summary": "s1"}]}
    as_dict = {"url": "u1", "article": {"document": ["d0", "d1"], "summary": ["s0", "s1"]}}
    a = standardize_wikilingua(as_list, "wikilingua_vi", "url")
    b = standardize_wikilingua(as_dict, "wikilingua_vi", "url")
    assert a == b
    assert [r["document"] for r in a] == ["d0", "d1"]
    assert set(a[0]) == {"id", "document", "summary"}
    assert a[0]["id"] != a[1]["id"]


def test_assign_split_deterministic_and_proportional():
    keys = [f"https://www.wikihow.com/item-{i}" for i in range(20_000)]
    first = [assign_split(k, 42, RATIOS) for k in keys]
    assert first == [assign_split(k, 42, RATIOS) for k in keys]
    frac = {s: c / len(keys) for s, c in Counter(first).items()}
    assert abs(frac["train"] - 0.9) < 0.01
    assert abs(frac["validation"] - 0.05) < 0.01
    assert abs(frac["test"] - 0.05) < 0.01
