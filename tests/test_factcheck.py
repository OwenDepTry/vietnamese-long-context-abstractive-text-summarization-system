"""Kiểm tra sự thật bề mặt (src/vnsum/factcheck.py) — ví dụ lấy từ các lỗi người chấm tìm thấy."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vnsum.factcheck import SourceIndex, check_summary, drop_unsupported, unsupported_items
from vnsum.inference.reduce import factcheck_mode, reduce_concat

ROOT = Path(__file__).resolve().parents[1]
SPLIT = lambda t: [s.strip() + "." for s in t.split(".") if s.strip()]  # noqa: E731

DOC = ("Công ty cổ phần Đầu tư bất động sản Mỹ Đình được cấp giấy chứng nhận đăng ký doanh nghiệp lần đầu ngày "
       "7/1/2015, vốn điều lệ 1.500 tỷ đồng. Theo luật sư Trần Đức Phượng, Đoàn Luật sư TP.HCM, dự án cần vay 420 tỷ. "
       "Đại hội Hội Luật gia TP. Hà Nội khóa VII, nhiệm kỳ 2018-2023. Tổng thống Mỹ Donald Trump phát biểu. "
       "Máy bay của hãng hàng không Qantas gặp nhiễu động.")


@pytest.mark.parametrize("sentence, expected", [
    ("Công ty Mỹ Đình được cấp giấy chứng nhận ngày 7/3/2015.", ["7/3/2015"]),            # sai ngày (H001)
    ("Theo luật sư Nguyễn Văn Phượng, dự án cần vay 420 tỷ.", ["Nguyễn Văn Phượng"]),     # đổi tên (H001)
    ("Đại hội Hội Luật gia TP. Hà Nội khóa VIII.", ["VIII"]),                             # sai khóa (H023)
])
def test_flags_numbers_and_names_not_in_source(sentence, expected):
    assert unsupported_items(sentence, SourceIndex(DOC)) == expected


@pytest.mark.parametrize("sentence", [
    "Công ty Mỹ Đình đăng ký ngày 7/1/2015 với vốn 1500 tỷ đồng.",   # cùng số, khác dấu phân cách hàng nghìn
    "Tổng thống Mỹ Donald Trump phát biểu.",                          # chuỗi viết hoa = "Mỹ" + "Donald Trump"
    "Nhận diện dấu hiệu sớm và trò chuyện thẳng thắn.",               # từ đầu câu viết hoa vì là đầu câu
    "Máy bay của hãng hàng không Qantas gặp nhiễu động.",
    "Luật sư Trần Đức Phượng cho biết dự án cần vay 420 tỷ.",
    "Ăn nhiều thực phẩm giàu vitamin B. Giảm lượng đường.",          # chữ cái đơn + đầu câu bị bộ tách câu bỏ sót
])
def test_does_not_flag_supported_sentences(sentence):
    assert unsupported_items(sentence, SourceIndex(DOC)) == []


def test_drop_unsupported_keeps_something():
    good, bad = "Luật sư Trần Đức Phượng phát biểu.", "Luật sư Nguyễn Văn Phượng phát biểu."
    assert drop_unsupported([good, bad], DOC) == [good]
    assert drop_unsupported([bad], DOC) == [bad]                        # bỏ hết thì giữ nguyên
    fc = check_summary(f"{good} {bad}", DOC, SPLIT)
    assert fc.flagged == 1 and fc.kept() == [good]


def test_reduce_concat_factcheck_drop():
    parts = ["Công ty Mỹ Đình đăng ký ngày 7/1/2015.", "Công ty Mỹ Đình đăng ký lại ngày 9/9/2019."]
    kw = dict(splitter=SPLIT, count_tokens=lambda t: len(t.split()), max_tokens=100, select=lambda t, n: t)
    assert "9/9/2019" in reduce_concat(parts, **kw)
    assert reduce_concat(parts, **kw, source=DOC, factcheck="drop") == parts[0]
    assert factcheck_mode(False) == "off"                               # YAML: off không ngoặc -> False
    with pytest.raises(ValueError):
        reduce_concat(parts, **kw, factcheck="fix")


def test_summarizer_returns_warnings(make_doc):
    from test_inference import FakeBackend
    from vnsum.config import load_config
    from vnsum.inference.config import load_inference_config
    from vnsum.inference.predictor import Summarizer

    cfg = load_inference_config(ROOT / "configs" / "inference.yaml")
    data_cfg = load_config(ROOT / "configs" / "data.yaml")

    class Echo(FakeBackend):
        def generate(self, texts, gk):
            out = super().generate(texts, gk)
            out.texts = ["Công ty Mỹ Đình đăng ký ngày 7/3/2015."] * len(texts)
            return out

    summ = Summarizer(Echo(), cfg, data_cfg)
    assert summ.summarize(DOC).unsupported == ["7/3/2015"]
    off = copy.deepcopy(cfg)
    off["factcheck_warnings"] = False
    assert Summarizer(Echo(), off, data_cfg).summarize(DOC).unsupported == []


def test_aux_metric():
    from vnsum.eval.metrics import aux_per_sample, unsupported_in

    row = aux_per_sample("Công ty Mỹ Đình đăng ký ngày 7/3/2015.", DOC, 3, [1], factcheck=True)
    assert row["unsupported_pct"] == 100.0 and unsupported_in("", DOC) == []
    assert "unsupported_pct" not in aux_per_sample("x", DOC, 3, [1])


def test_agrees_with_human_scores_on_long_v2():
    """Tái hiện đối chiếu với 40 mục người chấm. Quy tắc được chỉnh trên chính 40 mục này -> số lạc quan."""
    pd = pytest.importorskip("pandas")
    from vnsum.data.preprocess import split_sentences

    items, merged = (ROOT / "results/long_v2/human_eval" / f for f in ("items.md", "merged.csv"))
    if not (items.exists() and merged.exists()):
        pytest.skip("chưa có results/long_v2/human_eval")
    text = items.read_text(encoding="utf-8")
    m = pd.read_csv(merged, encoding="utf-8-sig")
    flagged = []
    for item in m["item_id"]:
        block = text[text.index(f"## {item}\n"):]
        block = block[: block.index("\n---")] if "\n---" in block else block
        doc, summ = block.split("**Bản tóm tắt cần chấm:**")
        doc = "\n".join(line[2:] if line.startswith("> ") else line for line in doc.splitlines())
        summ = summ.strip().split("\n")[0].lstrip("> ")
        flagged.append(check_summary(summ, doc, split_sentences).flagged > 0)
    m["flag"] = flagged
    assert m.loc[m["flag"] & (m["human_score"] >= 4)].empty            # không đánh dấu nhầm mục tốt
    assert m.loc[m["flag"], "human_score"].mean() < m.loc[~m["flag"], "human_score"].mean() - 1


def test_thousands_but_not_decimals():
    src = SourceIndex("Doanh thu đạt 1.500 tỷ đồng, tăng 3,5% so với năm 2018-2023.")
    assert unsupported_items("Doanh thu 1500 tỷ đồng.", src) == []
    assert unsupported_items("Doanh thu 1,500 tỷ đồng, nhiệm kỳ 2018.", src) == []
    assert unsupported_items("Tăng 35% so với năm trước.", src) == ["35"]


def test_factcheck_report_script(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    import importlib.util

    import vnsum.eval.dataset as dataset

    docs = pd.DataFrame({"id": ["a", "b"], "document": [DOC, DOC], "dataset": ["vietnews"] * 2, "is_long": [True] * 2})
    monkeypatch.setattr(dataset, "load_eval_set", lambda cfg, limit=None: docs)
    res = tmp_path / "res"
    (res / "human_eval").mkdir(parents=True)
    pd.DataFrame({
        "system": ["s1", "s1", "s2", "s2"], "id": ["a", "b", "a", "b"], "summary": ["Mỹ Đình đăng ký năm 2015."] * 4,
        "prediction": ["Mỹ Đình đăng ký ngày 7/3/2015.", "Mỹ Đình đăng ký ngày 7/1/2015.",
                       "Luật sư Nguyễn Văn Phượng nói.", "Luật sư Trần Đức Phượng nói."],
        "faithfulness": [1, 5, 2, None],
    }).to_csv(res / "per_sample.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"system": ["s1"], "id": ["a"], "human_score": [1]}).to_csv(
        res / "human_eval" / "merged.csv", index=False, encoding="utf-8-sig")
    spec = importlib.util.spec_from_file_location("factcheck_report", ROOT / "scripts" / "factcheck_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--config", "configs/eval.yaml", "--results_dir", str(res)]) == 0
    text = (res / "factcheck.md").read_text(encoding="utf-8")
    assert "| `s1` | 2 | 50.0 |" in text and "| `s2` | 2 | 50.0 |" in text
    assert "| Bị đánh dấu | 2 | 1.50 | 100.0 |" in text            # judge: 2 bản bị đánh dấu đều ≤ 2
    assert "7/3/2015" in text and "Nguyễn Văn Phượng" in text and "người chấm (độc lập" in text
