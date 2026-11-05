"""Bước reduce ``concat`` của tóm tắt phân cấp (src/vnsum/inference/reduce.py) — chạy offline."""

from __future__ import annotations

import copy

from pathlib import Path

import pytest

from test_inference import FakeBackend, cfg, data_cfg  # noqa: F401  (fixture + backend giả dùng lại)
from vnsum.eval.config import EvalConfigError, load_eval_config, validate_eval_config
from vnsum.inference.config import InferenceConfigError, validate_inference_config
from vnsum.inference.predictor import Summarizer
from vnsum.inference.reduce import dedupe_sentences, facts_compatible, overlap_ratio, reduce_concat

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
                                reduce="concat", chunk_length="medium", chunk_batch_size=1)
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
    ecfg = load_eval_config(Path(__file__).resolve().parents[1] / "configs" / "eval.yaml")
    sys_cfg = ecfg["systems"]["vit5_lora_hier_concat"]
    assert sys_cfg["enabled"] is False and sys_cfg["hierarchical"]["reduce"] == "concat"
    sys_cfg["hierarchical"]["reduce"] = "x"
    with pytest.raises(EvalConfigError):
        validate_eval_config(ecfg)


# --------------------------------------------------------------------------- dedupe: giống chữ ≠ giống thông tin
@pytest.mark.parametrize("a, b", [
    ("Năm 2023, doanh thu của công ty đạt 100 tỷ đồng.", "Năm 2024, doanh thu của công ty đạt 200 tỷ đồng."),
    ("Bộ Y tế đã xác nhận ca nhiễm mới tại Hà Nội.", "Bộ Y tế chưa xác nhận ca nhiễm mới tại Hà Nội."),
    ("Ông Huệ rời Bosch Việt Nam.", "Ông Huệ không rời Bosch Việt Nam."),
    ("Mưa lớn gây ngập nhiều tuyến đường ở Hà Nội hôm nay.", "Mưa lớn gây ngập nhiều tuyến đường ở Đà Nẵng hôm nay."),
    ("Chuyến bay cất cánh lúc 10:30 ngày 22/5.", "Chuyến bay cất cánh lúc 11:45 ngày 22/5."),
    # Review lần 2: tên riêng đứng đầu câu, và cùng tập số nhưng đảo quan hệ năm–số liệu.
    ("Nam đã nhận được học bổng toàn phần của trường đại học.", "Lan đã nhận được học bổng toàn phần của trường đại học."),
    ("Năm 2023 đạt 100 tỷ đồng, năm 2024 đạt 200 tỷ đồng.", "Năm 2023 đạt 200 tỷ đồng, năm 2024 đạt 100 tỷ đồng."),
])
def test_dedupe_keeps_sentences_with_different_facts(a, b):
    assert overlap_ratio(a, b) >= 0.6                      # giống chữ: trước đây bị gộp làm một
    assert dedupe_sentences([a, b]) == [a, b]


def test_dedupe_still_merges_true_duplicates():
    longer = "Năm 2023, doanh thu của công ty đạt 100 tỷ đồng, tăng mạnh so với năm trước."
    assert dedupe_sentences(["Năm 2023, doanh thu của công ty đạt 100 tỷ đồng.", longer]) == [longer]
    # Từ đầu câu viết hoa vẫn khớp khi câu được giữ chứa nó ở dạng thường.
    kept = "Theo báo cáo, năm 2023 doanh thu của công ty đạt 100 tỷ đồng."
    assert facts_compatible("Năm 2023 doanh thu của công ty đạt 100 tỷ đồng.", kept)
    assert dedupe_sentences(["Hà Nội mưa to.", "hà nội  mưa to ."]) == ["Hà Nội mưa to."]   # trùng chính xác


def test_compound_khong_is_not_negation():
    a = "Máy bay của hãng hàng không Qantas gặp nhiễu động."
    b = "Máy bay của hãng hàng không Qantas gặp nhiễu động mạnh trên không phận Australia."
    assert facts_compatible(a, b)
    assert dedupe_sentences([a, b]) == [b]


# --------------------------------------------------------------------------- batch + dùng lại tóm tắt chunk
def _concat_summarizer(cfg, data_cfg, **ld):  # noqa: F811
    cfg = copy.deepcopy(cfg)
    cfg["max_input_tokens"] = 40
    cfg["long_document"].update(chunk_max_tokens=30, overlap_tokens=5, hierarchical_above_tokens=100,
                                reduce="concat", chunk_length="medium", **ld)
    backend = FakeBackend()
    return Summarizer(backend, cfg, data_cfg), backend


def test_chunks_generated_in_batches(cfg, data_cfg, make_doc):  # noqa: F811
    summ, backend = _concat_summarizer(cfg, data_cfg, chunk_batch_size=4, chunk_cache_size=0)
    r = summ.summarize(make_doc(20)[0], length="short")
    n_chunks = len(r.chunk_summaries)
    assert n_chunks >= 5
    assert [len(c[0]) for c in backend.calls] == [min(4, n_chunks - i) for i in range(0, n_chunks, 4)]
    assert r.generate_calls == len(backend.calls) == -(-n_chunks // 4)


def test_three_lengths_reuse_chunk_summaries(cfg, data_cfg, make_doc):  # noqa: F811
    summ, backend = _concat_summarizer(cfg, data_cfg, chunk_batch_size=4, chunk_cache_size=4)
    doc = make_doc(20)[0]
    first = summ.summarize(doc, length="short")
    calls_after_first = len(backend.calls)
    others = [summ.summarize(doc, length=k) for k in ("medium", "long")]
    assert not first.chunk_cache_hit and all(r.chunk_cache_hit for r in others)
    assert len(backend.calls) == calls_after_first                     # không sinh lại chunk
    assert all(r.generate_calls == 0 and r.chunk_summaries == first.chunk_summaries for r in others)
    assert len(others[1].summary) >= len(first.summary)                # chỉ ngân sách chọn câu khác nhau
    summ.summarize(make_doc(21)[0], length="short")                    # văn bản khác: sinh mới
    assert len(backend.calls) > calls_after_first


def test_chunk_cache_is_bounded_and_can_be_disabled(cfg, data_cfg, make_doc):  # noqa: F811
    summ, backend = _concat_summarizer(cfg, data_cfg, chunk_cache_size=2)
    docs = [make_doc(n)[0] for n in (18, 19, 20)]
    for d in docs:
        summ.summarize(d, length="short")
    assert len(summ._chunk_cache) == 2
    assert not summ.summarize(docs[0], length="short").chunk_cache_hit   # đã bị đẩy ra (LRU)
    off, _ = _concat_summarizer(cfg, data_cfg, chunk_cache_size=0)
    off.summarize(docs[1], length="short")
    assert not off.summarize(docs[1], length="medium").chunk_cache_hit


def test_streaming_backend_forces_batch_one(cfg, data_cfg):  # noqa: F811
    from vnsum.api.streaming import StreamingBackend

    summ, _ = _concat_summarizer(cfg, data_cfg, chunk_batch_size=8)
    summ.backend = StreamingBackend(FakeBackend(), emit=lambda e: None, streamer_factory=lambda tok: iter(()))
    assert summ._chunk_batch_size() == 1
