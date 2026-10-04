import pytest

from vnsum.data.long_context import (
    LongContextProcessor,
    bm25_scores,
    chunk_document,
    extractive_filter,
    lexrank_scores,
)
from vnsum.data.preprocess import split_sentences


def _order_preserved(ids: list[int]) -> bool:
    return ids == sorted(ids) and len(set(ids)) == len(ids)


def _sentence_codes(text: str) -> list[int]:
    """Lấy chỉ số câu từ mã ``Sxx`` có trong văn bản đầu ra."""
    return [int(tok[1:]) for tok in text.split() if tok.startswith("S") and tok[1:].isdigit()]


# ------------------------------ xếp hạng ---------------------------------- #


@pytest.mark.parametrize("scorer", [bm25_scores, lexrank_scores])
def test_rankers_return_one_score_per_sentence(scorer, make_doc):
    _, sents = make_doc(10)
    scores = scorer(sents)
    assert scores.shape == (10,)
    assert (scores >= 0).all()


@pytest.mark.parametrize("scorer", [bm25_scores, lexrank_scores])
def test_rankers_prefer_central_sentence(scorer):
    sents = [
        "lạm phát tăng giá xăng tăng",
        "giá xăng tăng khiến lạm phát tăng mạnh",
        "lạm phát và giá xăng là mối lo",
        "con mèo ngủ trên mái nhà",
    ]
    scores = scorer(sents)
    assert scores.argmin() == 3  # câu lạc đề có điểm thấp nhất


# --------------------------- extractive_filter ----------------------------- #


@pytest.mark.parametrize("ranker", ["bm25", "lexrank"])
@pytest.mark.parametrize("max_tokens", [40, 100, 300])
def test_extractive_ratio_within_budget_and_in_order(counter, make_doc, ranker, max_tokens):
    text, sents = make_doc(40)  # ~13 token/câu -> ~520 token
    assert counter.count(text) > max_tokens
    res = extractive_filter(
        text, counter, max_input_tokens=max_tokens, ranker=ranker, selection="ratio", keep_ratio=0.3
    )
    # Tính cả special token: số token văn bản ≤ max - reserve.
    assert counter.count(res.text) + counter.reserve_special_tokens <= max_tokens
    assert res.num_tokens == counter.count(res.text)
    assert res.filtered
    assert _order_preserved(res.selected_ids)
    assert _order_preserved(_sentence_codes(res.text))  # thứ tự gốc thể hiện trong văn bản
    assert _sentence_codes(res.text) == res.selected_ids
    assert len(res.selected_ids) <= -(-40 * 3 // 10)  # ≤ ceil(30% * 40)


def test_extractive_keeps_top_k_when_budget_allows(counter, make_doc):
    text, _ = make_doc(20)
    res = extractive_filter(
        text, counter, max_input_tokens=10_000, selection="ratio", keep_ratio=0.3, only_if_exceeds=False
    )
    assert len(res.selected_ids) == 6  # ceil(0.3 * 20)
    assert _order_preserved(res.selected_ids)


@pytest.mark.parametrize("ranker", ["bm25", "lexrank"])
@pytest.mark.parametrize("max_tokens", [40, 100, 300])
def test_extractive_budget_fills_budget_in_order(counter, make_doc, ranker, max_tokens):
    text, sents = make_doc(40)
    budget = max_tokens - counter.reserve_special_tokens
    res = extractive_filter(text, counter, max_input_tokens=max_tokens, ranker=ranker, selection="budget")
    assert res.filtered
    assert counter.count(res.text) <= budget
    assert _order_preserved(res.selected_ids)
    assert _sentence_codes(res.text) == res.selected_ids
    # Đầy ngân sách: phần còn trống nhỏ hơn câu ngắn nhất chưa được chọn.
    left_out = [counter.count(s) for i, s in enumerate(sents) if i not in res.selected_ids]
    assert budget - res.num_tokens < min(left_out)


def test_extractive_budget_uses_more_context_than_ratio(counter, make_doc):
    text, _ = make_doc(40)  # ~520 token, chỉ vượt ngân sách 300 một chút
    budget_res = extractive_filter(text, counter, max_input_tokens=300, selection="budget")
    ratio_res = extractive_filter(text, counter, max_input_tokens=300, selection="ratio", keep_ratio=0.3)
    assert budget_res.num_tokens > ratio_res.num_tokens


def test_extractive_budget_skips_sentence_that_does_not_fit(counter):
    # Câu dài nhất được xếp hạng cao nhưng không vừa -> bỏ qua, vẫn nạp các câu ngắn.
    long_sent = "Lạm phát giá xăng tăng " + " ".join(["lạm phát giá xăng"] * 10) + "."
    short = [f"Lạm phát giá xăng tháng {i}." for i in range(5)]
    text = " ".join([short[0], long_sent, *short[1:]])
    res = extractive_filter(text, counter, max_input_tokens=26, selection="budget")
    assert 1 not in res.selected_ids
    assert len(res.selected_ids) == 4  # 4 câu x 6 token = 24 ≤ 25
    assert counter.count(res.text) <= 25


def test_extractive_invalid_selection(counter, make_doc):
    text, _ = make_doc(10)
    with pytest.raises(ValueError):
        extractive_filter(text, counter, max_input_tokens=50, selection="nope")


def test_extractive_short_document_untouched(counter, make_doc):
    text, _ = make_doc(3)
    res = extractive_filter(text, counter, max_input_tokens=1024, only_if_exceeds=True)
    assert res.text == text
    assert not res.filtered


def test_extractive_truncates_single_overlong_sentence(counter):
    text = "Một câu rất dài " + " ".join(f"w{i}" for i in range(200)) + "."
    res = extractive_filter(text, counter, max_input_tokens=50)
    assert res.truncated
    assert counter.count(res.text) <= 49


def test_extractive_empty(counter):
    res = extractive_filter("", counter, max_input_tokens=100)
    assert res.text == "" and res.selected_ids == []


# ------------------------------ chunking ----------------------------------- #


@pytest.mark.parametrize("chunk_max, overlap", [(60, 15), (100, 30), (200, 1)])
def test_chunking_no_sentence_lost_and_overlaps(counter, make_doc, chunk_max, overlap):
    text, _ = make_doc(30)
    sents = split_sentences(text)
    chunks = chunk_document(text, counter, chunk_max_tokens=chunk_max, overlap_tokens=overlap)
    assert len(chunks) > 1

    # 1) Không mất câu: mọi câu gốc có trong ít nhất một chunk, nguyên vẹn.
    covered = {sid for c in chunks for sid in c.sentence_ids}
    assert covered == set(range(len(sents)))
    for c in chunks:
        for sid in c.sentence_ids:
            assert sents[sid] in c.text

    # 2) Không cắt giữa câu: chunk là phép ghép các câu nguyên.
    for c in chunks:
        assert c.text == " ".join(sents[sid] for sid in c.sentence_ids)

    # 3) Có overlap giữa hai chunk liên tiếp và thứ tự câu được giữ.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.sentence_ids) & set(nxt.sentence_ids)
        assert nxt.sentence_ids[0] <= prev.sentence_ids[-1]
        assert _order_preserved(nxt.sentence_ids)
        assert nxt.sentence_ids[-1] > prev.sentence_ids[-1]  # luôn tiến lên

    # 4) Mỗi chunk ≤ giới hạn (tính cả special token).
    for c in chunks:
        assert counter.count(c.text) + counter.reserve_special_tokens <= chunk_max


def test_chunking_zero_overlap(counter, make_doc):
    text, _ = make_doc(20)
    chunks = chunk_document(text, counter, chunk_max_tokens=60, overlap_tokens=0)
    ids = [sid for c in chunks for sid in c.sentence_ids]
    assert ids == list(range(20))  # mỗi câu đúng một lần, đúng thứ tự


def test_chunking_overlong_sentence_is_split_but_kept(counter):
    long_sent = "Câu dài " + ", ".join(f"vế {i} có thêm vài từ" for i in range(30)) + "."
    text = f"Câu mở đầu ngắn. {long_sent} Câu kết thúc ngắn."
    chunks = chunk_document(text, counter, chunk_max_tokens=40, overlap_tokens=5)
    assert {sid for c in chunks for sid in c.sentence_ids} == {0, 1, 2}
    assert all(counter.count(c.text) <= 39 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    for i in range(30):
        assert f"vế {i} " in joined


def test_chunking_short_document_single_chunk(counter):
    chunks = chunk_document("Một câu. Hai câu.", counter, chunk_max_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1 and chunks[0].sentence_ids == [0, 1]


# --------------------------- processor theo config ------------------------- #

LC_CFG = {
    "strategy": "extractive_filter",
    "max_input_tokens": 64,
    "extractive_filter": {
        "ranker": "lexrank",
        "selection": "budget",
        "keep_ratio": 0.3,
        "min_sentences": 1,
        "only_if_exceeds": True,
        "bm25": {"k1": 1.5, "b": 0.75},
        "lexrank": {"threshold": 0.1, "damping": 0.85, "max_iter": 100, "tol": 1e-6},
    },
    "hierarchical": {"chunk_max_tokens": 50, "overlap_tokens": 10},
}


def test_processor_strategies(counter, make_doc):
    text, _ = make_doc(25)
    ex = LongContextProcessor(LC_CFG, counter, {"backend": "rule", "min_chars": 2})(text)
    assert ex["input_tokens"] <= 63 and ex["was_filtered"]
    hier = LongContextProcessor({**LC_CFG, "strategy": "hierarchical"}, counter, {"backend": "rule"})(text)
    assert hier["num_chunks"] == len(hier["chunks"]) > 1
    assert max(hier["chunk_tokens"]) <= 49
