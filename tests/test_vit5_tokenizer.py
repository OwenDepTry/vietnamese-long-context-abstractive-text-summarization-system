"""Kiểm tra ngân sách token với tokenizer ViT5 THẬT.

Tự skip nếu không tải được ``VietAI/vit5-base`` (máy offline hoặc thiếu ``transformers``).
"""

import pytest

from vnsum.data.long_context import chunk_document, extractive_filter
from vnsum.data.preprocess import split_sentences
from vnsum.data.tokenization import TokenCounter, load_tokenizer


@pytest.fixture(scope="module")
def vit5_counter():
    try:
        tok = load_tokenizer("VietAI/vit5-base", token_env="HF_TOKEN")
    except (ImportError, OSError) as exc:  # thiếu transformers hoặc không có mạng -> skip
        pytest.skip(f"Không tải được tokenizer ViT5: {exc}")
    # Lỗi khác (vd. TypeError do không tương thích version) phải làm test FAIL, không được skip.
    return TokenCounter(tok, reserve_special_tokens=1)


def _long_text(n: int = 120) -> str:
    base = [
        "Ngày 5/10, UBND TP. Hồ Chí Minh tổ chức họp báo về tình hình kinh tế - xã hội quý III.",
        "Theo báo cáo, tổng sản phẩm trên địa bàn tăng 7,2% so với cùng kỳ năm trước.",
        "Ông Nguyễn Văn A cho biết thành phố sẽ tập trung đầu tư hạ tầng giao thông và y tế.",
        "Nhiều doanh nghiệp bày tỏ lo ngại về giá nguyên vật liệu và chi phí logistics tăng cao.",
    ]
    return " ".join(f"{base[i % len(base)]}" for i in range(n))


def test_vit5_counts_special_tokens_separately(vit5_counter):
    tok = vit5_counter.tokenizer
    text = "Xin chào Việt Nam."
    with_special = len(tok(text)["input_ids"])
    assert with_special == vit5_counter.count(text) + vit5_counter.reserve_special_tokens


@pytest.mark.parametrize("ranker", ["bm25", "lexrank"])
def test_vit5_extractive_budget(vit5_counter, ranker):
    text = _long_text()
    assert vit5_counter.count(text) > 1024
    res = extractive_filter(text, vit5_counter, max_input_tokens=1024, ranker=ranker, keep_ratio=0.3)
    assert len(vit5_counter.tokenizer(res.text)["input_ids"]) <= 1024  # gồm cả </s>
    assert res.selected_ids == sorted(res.selected_ids)


def test_vit5_chunk_budget(vit5_counter):
    text = _long_text()
    chunks = chunk_document(text, vit5_counter, chunk_max_tokens=512, overlap_tokens=64)
    sents = split_sentences(text)
    assert {sid for c in chunks for sid in c.sentence_ids} == set(range(len(sents)))
    assert all(len(vit5_counter.tokenizer(c.text)["input_ids"]) <= 512 for c in chunks)
