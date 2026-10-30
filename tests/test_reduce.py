"""Bước reduce ``concat`` của tóm tắt phân cấp (src/vnsum/inference/reduce.py) — chạy offline."""

from __future__ import annotations

import copy

from pathlib import Path

import pytest

from test_inference import FakeBackend, cfg, data_cfg  # noqa: F401  (fixture + backend giả dùng lại)
from vnsum.eval.config import EvalConfigError, load_eval_config, validate_eval_config
from vnsum.inference.config import InferenceConfigError, validate_inference_config
from vnsum.inference.predictor import Summarizer
from vnsum.inference.reduce import dedupe_sentences, overlap_ratio, reduce_concat

SPLIT = lambda t: [s.strip() + "." for s in t.split(".") if s.strip()]  # noqa: E731


def test_overlap_ratio_containment():
    assert overlap_ratio("Ông Đủ bị điện giật.", "Ông Đủ bị điện giật khi hát karaoke.") == 1.0
    assert overlap_ratio("Hà Nội mưa to.", "Giá vàng tăng mạnh.") == 0.0
    # Giữ dấu: "Hà Nội" khác "Hè Nổi".
    assert overlap_ratio("Hà Nội mưa", "Hè Nổi mưa") < 0.6


def test_dedupe_keeps_order_and_longer_variant():
    out = dedupe_sentences([
        "Ông Đủ bị điện giật.",
        "Cảnh sát đã lập biên bản.",
        "Ông Đủ bị điện giật khi đang hát karaoke tại nhà.",
    ])
    assert out == ["Ông Đủ bị điện giật khi đang hát karaoke tại nhà.", "Cảnh sát đã lập biên bản."]


def test_reduce_concat_fits_without_selection():
    called = []
    out = reduce_concat(["Câu một về A. Câu hai về B.", "Câu hai về B. Câu ba về C."], splitter=SPLIT,
                        count_tokens=lambda t: len(t.split()), max_tokens=100,
                        select=lambda t, n: called.append(n) or t)
    assert out == "Câu một về A. Câu hai về B. Câu ba về C."
    assert not called                                      # vừa ngân sách -> không chọn câu


def test_reduce_concat_selects_when_too_long():
    topics = ["kinh tế tăng trưởng", "giáo dục đổi mới", "y tế quá tải", "thời tiết nắng nóng",
              "giao thông ùn tắc", "nông nghiệp được mùa", "du lịch phục hồi", "giá vàng biến động"]
    parts = [f"Bài viết nói về {t} trong năm nay." for t in topics]
    seen = []

    def select(text, n):
        seen.append(n)
        return " ".join(SPLIT(text)[:2])

    out = reduce_concat(parts, splitter=SPLIT, count_tokens=lambda t: len(t.split()), max_tokens=20, select=select)
    assert seen == [20] and out == " ".join(parts[:2])


def test_reduce_concat_ignores_empty_parts():
    out = reduce_concat(["", "  ", "Một câu."], splitter=SPLIT, count_tokens=lambda t: len(t.split()),
                        max_tokens=10, select=lambda t, n: t)
    assert out == "Một câu."


# --------------------------------------------------------------------------- Summarizer
def test_summarizer_concat_does_not_regenerate(cfg, data_cfg, make_doc):  # noqa: F811
    cfg = copy.deepcopy(cfg)
    cfg["max_input_tokens"] = 40
    cfg["long_document"].update(chunk_max_tokens=30, overlap_tokens=5, hierarchical_above_tokens=100,
                                reduce="concat", chunk_length="medium")
    backend = FakeBackend()
    summ = Summarizer(backend, cfg, data_cfg)
    doc, _ = make_doc(20)                                  # 240 từ > 100 -> hierarchical
    r = summ.summarize(doc, length="short")
    assert r.strategy == "hierarchical"
    assert r.generate_calls == len(backend.calls) == len(r.chunk_summaries) >= 2   # không có lần sinh cuối
    assert all(c[1]["max_new_tokens"] == 128 for c in backend.calls)               # chunk ở mức medium
    assert summ.counter.count(r.summary) <= 48                                     # ≤ max_new_tokens của short
    assert r.timings_ms["generate_ms"] == 0.0


def test_inference_config_rejects_bad_reduce(cfg):  # noqa: F811
    bad = copy.deepcopy(cfg)
    bad["long_document"]["reduce"] = "summarize_twice"
    with pytest.raises(InferenceConfigError):
        validate_inference_config(bad)


# --------------------------------------------------------------------------- eval
def test_hierarchical_concat_summaries_batches_all_chunks(counter, make_doc):
    from vnsum.eval.systems import hierarchical_concat_summaries

    calls = []

    def fake_generate(texts, n):
        calls.append((len(texts), n))
        return [f"Chunk {k} nói về ý {k}." for k in range(len(texts))]

    docs = [make_doc(30, 12)[0], make_doc(15, 12)[0]]
    out = hierarchical_concat_summaries(
        docs, fake_generate, counter, chunk_max_tokens=100, overlap_tokens=12, chunk_max_new_tokens=128,
        final_max_tokens=20, select=lambda t, n: " ".join(t.split()[:n]), splitter=SPLIT)
    assert len(calls) == 1 and calls[0][1] == 128          # mọi chunk của mọi văn bản: MỘT lần gọi
    assert len(out) == 2 and all(counter.count(o) <= 20 for o in out)


def test_eval_config_has_concat_system():
    cfg = load_eval_config(Path(__file__).resolve().parents[1] / "configs" / "eval.yaml")
    sys_cfg = cfg["systems"]["vit5_lora_hier_concat"]
    assert sys_cfg["enabled"] is False and sys_cfg["hierarchical"]["reduce"] == "concat"
    sys_cfg["hierarchical"]["reduce"] = "x"
    with pytest.raises(EvalConfigError):
        validate_eval_config(cfg)
