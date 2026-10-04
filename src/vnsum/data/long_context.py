"""Xử lý document dài hơn ngữ cảnh của ViT5.

Có hai chiến lược, chọn qua ``long_context.strategy``:

* ``extractive_filter``: tách câu, xếp hạng câu bằng BM25 hoặc LexRank, chọn
  câu theo ``selection`` (``budget``: nạp theo hạng tới khi đầy ngân sách token;
  ``ratio``: top-k câu, mặc định 30%), xuất theo THỨ TỰ GỐC. Đảm bảo số token
  ≤ ``max_input_tokens`` (tính cả special token).
* ``hierarchical``: chia document thành các chunk theo ranh giới câu, có
  overlap. Phase sau sẽ tóm tắt từng chunk rồi tóm tắt tổng.

Mọi phép đo độ dài đều qua ``TokenCounter`` (tokenizer ViT5), không đếm theo từ.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from vnsum.data.preprocess import split_sentences
from vnsum.data.tokenization import TokenCounter

_TERM_RE = re.compile(r"\w+")


def _terms(sentence: str) -> list[str]:
    return _TERM_RE.findall(sentence.lower())


# --------------------------------------------------------------------------- #
# Xếp hạng câu
# --------------------------------------------------------------------------- #


def bm25_scores(sentences: Sequence[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Độ trung tâm BM25: điểm của câu i = trung bình BM25(truy vấn = câu j, tài liệu = câu i), j ≠ i.

    Không có truy vấn ngoài, nên mỗi câu khác đóng vai truy vấn. Câu chia sẻ
    nhiều thuật ngữ hiếm với phần còn lại của document sẽ có điểm cao.
    """
    docs = [_terms(s) for s in sentences]
    n = len(docs)
    if n <= 1:
        return np.ones(n, dtype=float)
    lengths = np.array([len(d) for d in docs], dtype=float)
    avgdl = float(lengths.mean()) or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
    tfs = [Counter(d) for d in docs]

    scores = np.zeros(n, dtype=float)
    for i, tf_i in enumerate(tfs):
        norm = k1 * (1.0 - b + b * lengths[i] / avgdl)
        weight = {t: idf[t] * f * (k1 + 1.0) / (f + norm) for t, f in tf_i.items()}
        total = 0.0
        for j, tf_j in enumerate(tfs):
            if i != j:
                total += sum(weight.get(t, 0.0) * qf for t, qf in tf_j.items())
        scores[i] = total / (n - 1)
    return scores


def lexrank_scores(
    sentences: Sequence[str],
    threshold: float = 0.1,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """LexRank (Erkan & Radev 2004): đồ thị cosine TF-IDF có ngưỡng, rồi PageRank."""
    docs = [_terms(s) for s in sentences]
    n = len(docs)
    if n <= 1:
        return np.ones(n, dtype=float)
    vocab = {t: k for k, t in enumerate(sorted({t for d in docs for t in d}))}
    if not vocab:
        return np.ones(n, dtype=float) / n
    df = np.zeros(len(vocab))
    for d in docs:
        for t in set(d):
            df[vocab[t]] += 1
    idf = np.log(n / df) + 1.0
    mat = np.zeros((n, len(vocab)))
    for i, d in enumerate(docs):
        for t, f in Counter(d).items():
            mat[i, vocab[t]] = f * idf[vocab[t]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    sim = unit @ unit.T
    adj = (sim > threshold).astype(float)
    np.fill_diagonal(adj, 0.0)
    row_sum = adj.sum(axis=1, keepdims=True)
    # Câu cô lập: phân phối đều để ma trận vẫn ngẫu nhiên theo hàng.
    trans = np.where(row_sum > 0, adj / np.where(row_sum == 0, 1.0, row_sum), 1.0 / n)
    p = np.full(n, 1.0 / n)
    for _ in range(int(max_iter)):
        nxt = (1.0 - damping) / n + damping * (trans.T @ p)
        if np.abs(nxt - p).sum() < tol:
            p = nxt
            break
        p = nxt
    return p


def rank_sentences(sentences: Sequence[str], ranker: str, params: dict[str, Any]) -> np.ndarray:
    if ranker == "bm25":
        return bm25_scores(sentences, **params)
    if ranker == "lexrank":
        return lexrank_scores(sentences, **params)
    raise ValueError(f"ranker không hợp lệ: {ranker!r}")


# --------------------------------------------------------------------------- #
# Chiến lược a: extractive_filter
# --------------------------------------------------------------------------- #


@dataclass
class ExtractiveResult:
    text: str
    selected_ids: list[int]  # chỉ số câu được giữ, tăng dần (thứ tự gốc)
    num_sentences: int
    num_tokens: int
    filtered: bool  # False nếu document đã vừa giới hạn và được giữ nguyên
    truncated: bool = False  # True nếu phải cắt cả câu tốt nhất (câu đơn lẻ quá dài)


def extractive_filter(
    text: str,
    counter: TokenCounter,
    *,
    max_input_tokens: int,
    ranker: str = "bm25",
    selection: str = "budget",
    keep_ratio: float = 0.3,
    min_sentences: int = 1,
    only_if_exceeds: bool = True,
    ranker_params: dict[str, Any] | None = None,
    splitter: Callable[[str], list[str]] = split_sentences,
) -> ExtractiveResult:
    """Chọn câu theo điểm, xuất theo thứ tự gốc, đảm bảo ≤ ``max_input_tokens``.

    ``selection``:
      * ``budget``: nạp câu theo thứ hạng giảm dần cho tới khi đầy ngân sách
        token (câu không vừa thì bỏ qua, thử câu kế tiếp). Dùng tối đa ngữ cảnh.
      * ``ratio``: giữ top ``keep_ratio`` số câu (tối thiểu ``min_sentences``),
        rồi bớt câu điểm thấp nhất nếu vẫn vượt ngân sách.
    """
    if selection not in ("budget", "ratio"):
        raise ValueError(f"selection không hợp lệ: {selection!r}")
    budget = counter.budget(max_input_tokens)
    sentences = splitter(text)
    n = len(sentences)
    if n == 0:
        return ExtractiveResult("", [], 0, 0, filtered=False)

    full_tokens = counter.count(text)
    if only_if_exceeds and full_tokens <= budget:
        return ExtractiveResult(text, list(range(n)), n, full_tokens, filtered=False)

    scores = rank_sentences(sentences, ranker, ranker_params or {})
    # Sắp giảm dần theo điểm; hòa điểm thì câu đứng trước được ưu tiên.
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    sent_tokens = counter.count_many(sentences)

    if selection == "budget":
        selected: list[int] = []
        total = 0
        for i in order:  # selected giữ thứ tự theo hạng (cao -> thấp)
            if total + sent_tokens[i] <= budget:
                selected.append(i)
                total += sent_tokens[i]
    else:
        k = min(n, max(int(min_sentences), math.ceil(float(keep_ratio) * n)))
        selected = order[:k]
        while len(selected) > 1 and sum(sent_tokens[i] for i in selected) > budget:
            selected.pop()

    # Kiểm tra chính xác trên văn bản ghép (token tại ranh giới câu có thể khác
    # tổng từng câu); vượt thì bớt câu hạng thấp nhất.
    while selected:
        ids = sorted(selected)
        out = " ".join(sentences[i] for i in ids)
        n_tok = counter.count(out)
        if n_tok <= budget:
            return ExtractiveResult(out, ids, n, n_tok, filtered=True)
        if len(selected) == 1:
            break
        selected.pop()

    # Câu tốt nhất vẫn vượt ngân sách -> cắt theo token.
    best = order[0]
    out = counter.truncate(sentences[best], budget)
    return ExtractiveResult(out, [best], n, counter.count(out), filtered=True, truncated=True)


# --------------------------------------------------------------------------- #
# Chiến lược b: hierarchical chunking
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    text: str
    sentence_ids: list[int]  # chỉ số câu gốc; câu bị chia nhỏ xuất hiện ở nhiều chunk
    num_tokens: int
    overlap_units: int = 0  # số đơn vị (câu) đầu chunk lặp lại từ chunk trước


@dataclass
class _Unit:
    sid: int
    text: str
    ntok: int


_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")


def _units_from_sentences(sentences: Sequence[str], counter: TokenCounter, budget: int) -> list[_Unit]:
    """Mỗi câu là một đơn vị. Câu dài hơn cả chunk thì chia theo dấu phẩy/chấm phẩy,
    cuối cùng mới cắt theo token. Đây là ngoại lệ duy nhất được phép cắt giữa câu."""
    counts = counter.count_many(list(sentences))
    units: list[_Unit] = []
    for sid, (sent, ntok) in enumerate(zip(sentences, counts)):
        if ntok <= budget:
            units.append(_Unit(sid, sent, ntok))
            continue
        pieces: list[str] = []
        buf = ""
        for clause in _CLAUSE_RE.split(sent):
            cand = f"{buf} {clause}".strip()
            if buf and counter.count(cand) > budget:
                pieces.append(buf)
                buf = clause
            else:
                buf = cand
        if buf:
            pieces.append(buf)
        for piece in pieces:
            while counter.count(piece) > budget:
                head = counter.truncate(piece, budget)
                if not head:
                    break
                units.append(_Unit(sid, head, counter.count(head)))
                piece = piece[len(head) :].strip()
            if piece:
                units.append(_Unit(sid, piece, counter.count(piece)))
    return units


def chunk_document(
    text: str,
    counter: TokenCounter,
    *,
    chunk_max_tokens: int,
    overlap_tokens: int,
    splitter: Callable[[str], list[str]] = split_sentences,
) -> list[Chunk]:
    """Chia document thành chunk theo ranh giới câu, mỗi chunk ≤ ``chunk_max_tokens``.

    Overlap: chunk mới bắt đầu bằng các câu cuối của chunk trước, tổng tối đa
    ``overlap_tokens`` (ít nhất 1 câu nếu ``overlap_tokens > 0``). Nếu câu kế
    tiếp quá dài để vừa cùng phần overlap, phần overlap được bớt dần từ đầu.
    """
    budget = counter.budget(chunk_max_tokens)
    sentences = splitter(text)
    if not sentences:
        return []
    units = _units_from_sentences(sentences, counter, budget)

    def join(idx: list[int]) -> str:
        return " ".join(units[i].text for i in idx)

    def make_chunk(idx: list[int], n_overlap: int) -> Chunk:
        sids: list[int] = []
        for i in idx:
            if not sids or sids[-1] != units[i].sid:
                sids.append(units[i].sid)
        body = join(idx)
        return Chunk(body, sids, counter.count(body), overlap_units=n_overlap)

    def overlap_tail(idx: list[int]) -> list[int]:
        if overlap_tokens <= 0 or len(idx) <= 1:
            return []
        tail: list[int] = []
        total = 0
        for i in reversed(idx[1:]):  # không bao giờ lặp lại toàn bộ chunk
            if tail and total + units[i].ntok > overlap_tokens:
                break
            tail.insert(0, i)
            total += units[i].ntok
        return tail

    chunks: list[Chunk] = []
    cur: list[int] = []
    n_carried = 0
    i = 0
    while i < len(units):
        cand = cur + [i]
        if counter.count(join(cand)) <= budget:
            cur = cand
            i += 1
            continue
        if len(cur) > n_carried:  # chunk hiện tại có nội dung mới -> chốt
            chunks.append(make_chunk(cur, n_carried))
            cur = overlap_tail(cur)
            n_carried = len(cur)
        elif cur:  # chỉ còn overlap mà vẫn không vừa -> bớt overlap từ đầu
            cur.pop(0)
            n_carried -= 1
        else:  # đơn vị đơn lẻ (đã đảm bảo ≤ budget khi tạo unit)
            cur = [i]
            i += 1
    if len(cur) > n_carried:
        chunks.append(make_chunk(cur, n_carried))
    return chunks


# --------------------------------------------------------------------------- #
# Áp dụng theo config
# --------------------------------------------------------------------------- #


@dataclass
class LongContextProcessor:
    """Bọc hai chiến lược với tham số lấy từ ``cfg['long_context']``."""

    cfg: dict[str, Any]
    counter: TokenCounter
    sentence_cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strategy = self.cfg["strategy"]
        backend = self.sentence_cfg.get("backend", "rule")
        min_chars = int(self.sentence_cfg.get("min_chars", 2))
        self.splitter = lambda t: split_sentences(t, backend=backend, min_chars=min_chars)

    def __call__(self, document: str) -> dict[str, Any]:
        if self.strategy == "extractive_filter":
            ex = self.cfg["extractive_filter"]
            ranker = ex["ranker"]
            res = extractive_filter(
                document,
                self.counter,
                max_input_tokens=int(self.cfg["max_input_tokens"]),
                ranker=ranker,
                selection=ex["selection"],
                keep_ratio=float(ex["keep_ratio"]),
                min_sentences=int(ex["min_sentences"]),
                only_if_exceeds=bool(ex["only_if_exceeds"]),
                ranker_params=dict(ex.get(ranker, {})),
                splitter=self.splitter,
            )
            return {
                "input_text": res.text,
                "input_tokens": res.num_tokens,
                "selected_sentence_ids": res.selected_ids,
                "num_sentences": res.num_sentences,
                "was_filtered": res.filtered,
                "was_truncated": res.truncated,
            }
        if self.strategy == "hierarchical":
            h = self.cfg["hierarchical"]
            chunks = chunk_document(
                document,
                self.counter,
                chunk_max_tokens=int(h["chunk_max_tokens"]),
                overlap_tokens=int(h["overlap_tokens"]),
                splitter=self.splitter,
            )
            return {
                "chunks": [c.text for c in chunks],
                "chunk_tokens": [c.num_tokens for c in chunks],
                "chunk_sentence_ids": [c.sentence_ids for c in chunks],
                "num_chunks": len(chunks),
            }
        raise ValueError(f"long_context.strategy không hợp lệ: {self.strategy!r}")
