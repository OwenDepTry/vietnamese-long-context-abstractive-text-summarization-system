"""scripts/length_control.py — đối chứng độ dài, chạy offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("length_control", ROOT / "scripts" / "length_control.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


def test_cut_syllables_and_sentences():
    text = "Hà Nội mưa to. Đường phố ngập nặng. Giao thông tê liệt."
    assert lc.cut_syllables(text, 4) == "Hà Nội mưa to."
    assert lc.n_syllables(lc.cut_syllables(text, 6)) == 6
    assert lc.cut_sentences(text, 9) == "Hà Nội mưa to. Đường phố ngập nặng."
    assert lc.cut_sentences(text, 1) == "Hà Nội mưa to."          # luôn giữ ít nhất một câu


def test_matched_only_shortens():
    long, short = "một hai ba bốn năm sáu.", "một hai."
    assert lc.matched([long, short], [3, 10], "syllables") == ["một hai ba", short]


def test_report_detects_pure_length_advantage():
    pytest.importorskip("rouge_score")
    ref = "một hai ba bốn năm sáu bảy tám"
    rows = []
    for k in range(6):
        rows.append({"system": "base", "id": str(k), "summary": ref, "prediction": "một hai ba"})
        # Cùng nội dung, chỉ dài hơn: cắt về cùng độ dài thì hiệu phải về 0.
        rows.append({"system": "long", "id": str(k), "summary": ref, "prediction": "một hai ba bốn năm sáu"})
    text = lc.build_report(pd.DataFrame(rows), "long", ["base"], n_boot=200, seed=0)
    assert "cắt theo âm tiết | 3.0 |" in text and "+0.00 [+0.00, +0.00]" in text
