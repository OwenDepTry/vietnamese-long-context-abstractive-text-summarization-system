"""Các hệ thống tóm tắt được so sánh + cache dự đoán ra file (không sinh lại khi chạy lại).

* ``lead``: n câu đầu của document (baseline extractive).
* ``seq2seq``: model ViT5 (LoRA đã merge, hoặc checkpoint công khai) với generation config chung.
* ``llm``: zero-shot qua API (chỉ khi --allow_api).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from vnsum.data.preprocess import split_sentences
from vnsum.eval.llm_client import ApiGate, LLMClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cache dự đoán
# --------------------------------------------------------------------------- #


def fingerprint(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


def input_hash(text: str) -> str:
    """Hash nội dung input của MỘT mẫu (sau tiền xử lý), lưu kèm từng dự đoán trong cache."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


class PredictionCache:
    """``<dir>/<system>.jsonl``: mỗi dòng ``{id, prediction, fp, in}``.

    Một dòng chỉ được dùng lại khi cùng fingerprint hệ thống (``fp``: config + checkpoint + phiên bản pipeline)
    VÀ cùng hash nội dung input (``in``) của mẫu đó. Trước đây khóa chỉ là (fp, id): sửa dữ liệu mà giữ nguyên
    id thì cache vẫn trả dự đoán cũ.

    ``accept_legacy``: dòng cũ không có ``in`` (ghi trước khi có kiểm tra này) có được dùng lại không. Chỉ nên
    bật cho hệ thống gọi API trả phí; số dòng như vậy được ghi vào ``legacy_used`` để cảnh báo.
    """

    def __init__(self, directory: str | Path, system: str, fp: str, *, accept_legacy: bool = False) -> None:
        self.path = Path(directory) / f"{system}.jsonl"
        self.fp = fp
        self.accept_legacy = accept_legacy
        self._lock = threading.Lock()
        self.data: dict[str, str] = {}
        self._in: dict[str, str | None] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("fp") == fp:
                        self.data[str(row["id"])] = row["prediction"]
                        self._in[str(row["id"])] = row.get("in")
        self.legacy_used = 0

    def _valid(self, i: str, h: str | None) -> bool:
        if i not in self.data:
            return False
        if h is None:
            return True
        stored = self._in.get(i)
        if stored is None:
            return self.accept_legacy
        return stored == h

    def missing(self, ids: Sequence[str], input_hashes: Sequence[str] | None = None) -> list[str]:
        hashes = list(input_hashes) if input_hashes is not None else [None] * len(ids)
        out = [str(i) for i, h in zip(ids, hashes) if not self._valid(str(i), h)]
        if input_hashes is not None:
            self.legacy_used = sum(1 for i in ids if str(i) not in out and self._in.get(str(i)) is None)
        return out

    def add(self, items: dict[str, str], input_hashes: dict[str, str] | None = None) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for i, pred in items.items():
                    i = str(i)
                    h = (input_hashes or {}).get(i)
                    self.data[i] = pred
                    self._in[i] = h
                    row = {"id": i, "prediction": pred, "fp": self.fp}
                    if h is not None:
                        row["in"] = h
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Lead-n
# --------------------------------------------------------------------------- #


def lead_n(document: str, n: int = 3) -> str:
    return " ".join(split_sentences(document)[:n])


# --------------------------------------------------------------------------- #
# Seq2seq (ViT5)
# --------------------------------------------------------------------------- #


def resolve_dtype(name: str):
    import torch

    if name == "fp32" or not torch.cuda.is_available():
        return torch.float32
    if name in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name == "bf16":
        raise SystemExit(f"GPU {torch.cuda.get_device_name(0)} không hỗ trợ bf16; đặt generation.dtype: fp32")
    return torch.float32


class Seq2SeqGenerator:
    """Nạp model seq2seq MỘT lần, sinh theo batch với generation config chung của phase 3."""

    def __init__(self, model_path: str, sys_cfg: dict[str, Any], gen: dict[str, Any], token_env: str | None) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM

        from vnsum.data.tokenization import load_tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = resolve_dtype(gen.get("dtype", "auto"))
        token = os.environ.get(token_env) if token_env else None
        logger.info("Load %s (%s, %s)", model_path, self.device, dtype)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=token or None, dtype=dtype).to(self.device).eval()
        self.tok = load_tokenizer(model_path, token_env)
        self.name = Path(str(model_path)).name
        # sys_cfg.generation (tùy chọn) ghi đè generation chung cho riêng hệ thống này, vd. tham số độ dài đã
        # tune ở bước 2 (scripts/tune_lengths.py) mà không làm đổi các hệ thống khác.
        self.sys_cfg, self.gen = sys_cfg, effective_generation({"generation": gen}, sys_cfg)

    def generate(self, texts: Sequence[str], *, max_new_tokens: int | None = None, desc: str | None = None) -> list[str]:
        import torch

        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            def tqdm(x, **_):
                return x

        sys_cfg, gen = self.sys_cfg, self.gen
        prefix = sys_cfg.get("source_prefix") or ""
        bs = int(gen["batch_size"])
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # gom input cùng độ dài -> ít padding
        preds: list[str] = [""] * len(texts)
        for start in tqdm(range(0, len(order), bs), desc=desc or f"generate {self.name}"):
            idx = order[start : start + bs]
            enc = self.tok(
                [prefix + texts[i] for i in idx],
                max_length=int(sys_cfg["max_source_length"]),
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    num_beams=int(gen["num_beams"]),
                    max_new_tokens=int(max_new_tokens or gen["max_new_tokens"]),
                    no_repeat_ngram_size=int(gen["no_repeat_ngram_size"]),
                    early_stopping=bool(gen.get("early_stopping", True)),
                    **_length_kwargs(gen, int(max_new_tokens or gen["max_new_tokens"])),
                )
            for i, text in zip(idx, self.tok.batch_decode(out, skip_special_tokens=True)):
                preds[i] = text.strip()
        return preds

    def close(self) -> None:
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _length_kwargs(gen: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    """min_new_tokens / length_penalty tùy chọn; chỉ truyền khi khác mặc định để giữ nguyên hành vi cũ.

    min_new_tokens bị bỏ khi lần sinh có max_new_tokens nhỏ hơn (vd. tóm tắt chunk ngắn ở mức hierarchical).
    """
    out: dict[str, Any] = {}
    mn = int(gen.get("min_new_tokens") or 0)
    if 0 < mn < int(max_new_tokens):
        out["min_new_tokens"] = mn
    lp = float(gen.get("length_penalty", 1.0))
    if lp != 1.0 and int(gen["num_beams"]) > 1:
        out["length_penalty"] = lp
    return out


def generate_seq2seq(texts: Sequence[str], model_path: str, sys_cfg: dict[str, Any], gen: dict[str, Any], token_env: str | None) -> list[str]:
    g = Seq2SeqGenerator(model_path, sys_cfg, gen, token_env)
    try:
        return g.generate(texts)
    finally:
        g.close()


# --------------------------------------------------------------------------- #
# Tóm tắt phân cấp cho văn bản dài (cùng thuật toán với Summarizer phase 4)
# --------------------------------------------------------------------------- #


def hierarchical_inputs(
    docs: Sequence[str],
    generate,
    counter,
    *,
    max_input_tokens: int,
    chunk_max_tokens: int,
    overlap_tokens: int,
    chunk_max_new_tokens: int,
    max_rounds: int,
    extract,
    splitter,
) -> tuple[list[str], list[int]]:
    """Trả (input cho lần sinh cuối, số vòng chia chunk) cho từng văn bản.

    Mỗi vòng: chia văn bản còn quá dài thành chunk (≤ ``chunk_max_tokens``, overlap theo câu), tóm tắt mọi
    chunk của mọi văn bản trong MỘT lần gọi ``generate`` (batch), ghép các bản tóm tắt. Dừng khi vừa
    ``max_input_tokens`` hoặc hết ``max_rounds``; còn quá dài thì lọc trích xuất như lúc train.
    ``generate(texts, max_new_tokens)`` -> list[str].
    """
    from vnsum.data.long_context import chunk_document

    budget = counter.budget(int(max_input_tokens))
    current = list(docs)
    rounds = [0] * len(current)
    for _ in range(int(max_rounds)):
        todo = [i for i, t in enumerate(current) if counter.count(t) > budget]
        if not todo:
            break
        owners, chunk_texts = [], []
        for i in todo:
            for c in chunk_document(current[i], counter, chunk_max_tokens=int(chunk_max_tokens),
                                    overlap_tokens=int(overlap_tokens), splitter=splitter):
                owners.append(i)
                chunk_texts.append(c.text)
        parts = generate(chunk_texts, int(chunk_max_new_tokens))
        joined: dict[int, list[str]] = {i: [] for i in todo}
        for i, part in zip(owners, parts):
            if part:
                joined[i].append(part)
        for i in todo:
            current[i] = " ".join(joined[i])
            rounds[i] += 1
    return [t if counter.count(t) <= budget else extract(t) for t in current], rounds


def hierarchical_concat_summaries(
    docs: Sequence[str],
    generate,
    counter,
    *,
    chunk_max_tokens: int,
    overlap_tokens: int,
    chunk_max_new_tokens: int,
    final_max_tokens: int,
    select,
    splitter,
    dedupe_threshold: float = 0.6,
) -> list[str]:
    """Reduce ``concat``: tóm tắt mọi chunk của mọi văn bản trong MỘT lần gọi ``generate`` (batch), rồi
    ghép + bỏ câu trùng + chọn câu (≤ ``final_max_tokens``) — không sinh lại. Trả bản tóm tắt cuối.
    ``select(text, max_tokens)`` -> str; ``generate(texts, max_new_tokens)`` -> list[str].
    """
    from vnsum.data.long_context import chunk_document
    from vnsum.inference.reduce import reduce_concat

    owners, chunk_texts = [], []
    for i, doc in enumerate(docs):
        for c in chunk_document(doc, counter, chunk_max_tokens=int(chunk_max_tokens),
                                overlap_tokens=int(overlap_tokens), splitter=splitter):
            owners.append(i)
            chunk_texts.append(c.text)
    parts = generate(chunk_texts, int(chunk_max_new_tokens)) if chunk_texts else []
    per_doc: list[list[str]] = [[] for _ in docs]
    for i, part in zip(owners, parts):
        per_doc[i].append(part)
    return [reduce_concat(p, splitter=splitter, count_tokens=counter.count, max_tokens=int(final_max_tokens),
                          select=select, dedupe_threshold=float(dedupe_threshold)) for p in per_doc]


def run_hierarchical_seq2seq(df, model_path: str, sys_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Văn bản dài (``is_long``) -> tóm tắt phân cấp từ document gốc; văn bản ngắn -> ``input_column`` như thường."""
    from vnsum.config import load_config
    from vnsum.data.long_context import extractive_filter
    from vnsum.data.preprocess import split_sentences as _split
    from vnsum.data.tokenization import TokenCounter

    h = sys_cfg["hierarchical"]
    data_cfg = load_config(h.get("data_config", "configs/data.yaml"))
    sc = data_cfg.get("sentence_split", {}) or {}
    splitter = lambda t: _split(t, backend=sc.get("backend", "rule"), min_chars=int(sc.get("min_chars", 2)))  # noqa: E731
    ex = data_cfg["long_context"]["extractive_filter"]
    max_in = int(sys_cfg["max_source_length"])
    g = Seq2SeqGenerator(model_path, sys_cfg, cfg["generation"], cfg.get("hf", {}).get("token_env"))
    try:
        counter = TokenCounter(g.tok, int(data_cfg["tokenizer"].get("reserve_special_tokens", 1)))

        def extract(t: str, max_tokens: int | None = None) -> str:
            limit = max_in if max_tokens is None else int(max_tokens) + counter.reserve_special_tokens
            return extractive_filter(t, counter, max_input_tokens=limit, ranker=ex["ranker"], selection=ex["selection"],
                                     keep_ratio=float(ex["keep_ratio"]), min_sentences=int(ex["min_sentences"]),
                                     only_if_exceeds=bool(ex["only_if_exceeds"]), ranker_params=dict(ex.get(ex["ranker"], {})),
                                     splitter=splitter).text

        is_long = df["is_long"].astype(bool).tolist()
        inputs = df[sys_cfg["input_column"]].fillna("").tolist()
        long_idx = [k for k, v in enumerate(is_long) if v]
        if long_idx and h.get("reduce", "generate") == "concat":
            long_docs = df["document"].fillna("").iloc[long_idx].tolist()
            summaries = hierarchical_concat_summaries(
                long_docs, lambda texts, n: g.generate(texts, max_new_tokens=n, desc="hierarchical: chunk"),
                counter, chunk_max_tokens=int(h["chunk_max_tokens"]), overlap_tokens=int(h["overlap_tokens"]),
                chunk_max_new_tokens=int(h["chunk_max_new_tokens"]),
                final_max_tokens=int(h.get("final_max_tokens", cfg["generation"]["max_new_tokens"])),
                select=extract, splitter=splitter, dedupe_threshold=float(h.get("dedupe_threshold", 0.6)),
            )
            short_idx = [k for k, v in enumerate(is_long) if not v]
            preds = [""] * len(inputs)
            if short_idx:
                for k, p in zip(short_idx, g.generate([inputs[k] for k in short_idx], desc="hierarchical: short")):
                    preds[k] = p
            for k, p in zip(long_idx, summaries):
                preds[k] = p
            logger.info("hierarchical (concat): %d văn bản dài", len(long_idx))
            return preds
        if long_idx:
            final, rounds = hierarchical_inputs(
                df["document"].fillna("").iloc[long_idx].tolist(),
                lambda texts, n: g.generate(texts, max_new_tokens=n, desc="hierarchical: chunk"),
                counter, max_input_tokens=max_in, chunk_max_tokens=int(h["chunk_max_tokens"]),
                overlap_tokens=int(h["overlap_tokens"]), chunk_max_new_tokens=int(h["chunk_max_new_tokens"]),
                max_rounds=int(h["max_rounds"]), extract=extract, splitter=splitter,
            )
            for k, t in zip(long_idx, final):
                inputs[k] = t
            logger.info("hierarchical: %d văn bản dài, số vòng chia chunk %s", len(long_idx),
                        {r: rounds.count(r) for r in sorted(set(rounds))})
        return g.generate(inputs, desc="hierarchical: final")
    finally:
        g.close()


# --------------------------------------------------------------------------- #
# Pipeline triển khai (Summarizer của API)
# --------------------------------------------------------------------------- #


def _inference_config(sys_cfg: dict[str, Any]) -> dict[str, Any]:
    from vnsum.inference.config import load_inference_config

    icfg = load_inference_config(sys_cfg["inference_config"])
    for key, value in (sys_cfg.get("long_document") or {}).items():   # override tùy chọn, vd. chunk_batch_size
        icfg["long_document"][key] = value
    return icfg


def generate_pipeline(texts: Sequence[str], model_path: str, sys_cfg: dict[str, Any]) -> list[str]:
    """Chạy ĐÚNG đường xử lý của API: ``Summarizer`` + ``configs/inference.yaml`` trên văn bản gốc, từng văn bản
    một như request thật (chuẩn hóa, chọn chiến lược, sinh chunk theo ``chunk_batch_size`` TRONG văn bản, reduce
    concat). Khác ``seq2seq``: eval tự lọc câu và sinh theo batch gom nhiều văn bản."""
    from vnsum.inference.predictor import Summarizer

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    icfg = _inference_config(sys_cfg)
    summ = Summarizer.from_config(icfg, sys_cfg.get("backend", "pytorch"), model_path=model_path,
                                  device=sys_cfg.get("device"))
    length = sys_cfg.get("length", icfg["default_length"])
    out = []
    for t in tqdm(list(texts), desc=f"pipeline ({summ.backend.name} {summ.backend.dtype}, {length})"):
        out.append(summ.summarize(t, length=length).summary if t.strip() else "")
    return out


# --------------------------------------------------------------------------- #
# LLM zero-shot
# --------------------------------------------------------------------------- #


def build_llm_prompt(template: str, document: str, max_chars: int) -> str:
    return template.replace("{document}", document[: int(max_chars)])


def generate_llm(
    ids: Sequence[str], texts: Sequence[str], client: LLMClient, template: str, max_chars: int, cache: PredictionCache,
    input_hashes: dict[str, str] | None = None,
) -> None:
    """Gọi API cho từng văn bản; ghi cache ngay sau mỗi request (dừng giữa chừng không mất phần đã trả phí)."""

    def _one(pair: tuple[str, str]) -> None:
        i, doc = pair
        text, _ = client.complete(build_llm_prompt(template, doc, max_chars))
        cache.add({i: text}, input_hashes)

    client.map(list(zip(ids, texts)), _one)


# --------------------------------------------------------------------------- #
# Điểm vào: chạy (hoặc lấy cache) cho một hệ thống
# --------------------------------------------------------------------------- #


# Tăng khi đổi code sinh / reduce / tiền xử lý của hệ thống seq2seq mà config không đổi
# (vd. 2: sửa dedupe của reduce concat) -> cache dự đoán cũ không bị dùng lại.
PIPELINE_VERSION = 4   # 3: dedupe xét tên riêng đầu câu và thứ tự số liệu; 4: fingerprint gồm batch_size

_MODEL_HASHES: dict[tuple, str] = {}
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def model_identity(model_path: str | None) -> str | None:
    """Danh tính checkpoint: thư mục model trên máy -> hash nội dung config + trọng số (không phụ thuộc đường
    dẫn, nên chuyển thư mục model không làm mất cache, còn ghi đè trọng số thì có). Model trên HF Hub -> giữ
    nguyên tên (không kiểm tra revision).

    Hash được nhớ trong tiến trình theo (đường dẫn, tên + kích thước + thời điểm sửa của từng file), nên ghi đè
    trọng số ngay trong cùng tiến trình cũng làm hash được tính lại.
    """
    if not model_path:
        return model_path
    p = Path(model_path)
    if not p.is_dir():
        return model_path
    files = sorted(f for f in p.iterdir() if f.is_file() and (f.name == "config.json" or f.suffix in _WEIGHT_SUFFIXES))
    if not files:
        return model_path
    key = (str(p.resolve()), tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files))
    if key not in _MODEL_HASHES:
        h = hashlib.sha256()
        for f in files:
            h.update(f.name.encode("utf-8"))
            with f.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
        _MODEL_HASHES[key] = "sha256:" + h.hexdigest()[:16]
    return _MODEL_HASHES[key]


def effective_generation(cfg: dict[str, Any], sys_cfg: dict[str, Any]) -> dict[str, Any]:
    """generation chung của eval.yaml + ``systems.<tên>.generation`` (giống cách ``Seq2SeqGenerator`` áp dụng)."""
    return {**cfg["generation"], **(sys_cfg.get("generation") or {})}


def system_fingerprint(name: str, sys_cfg: dict[str, Any], cfg: dict[str, Any], model_path: str | None) -> str:
    # display_name/description chỉ ảnh hưởng report -> không làm mất cache dự đoán khi đổi tên.
    shown = {k: v for k, v in sys_cfg.items() if k not in ("display_name", "description")}
    payload: dict[str, Any] = {"name": name, "system": shown, "model_path": model_path}
    if sys_cfg["type"] == "seq2seq":
        payload["model_path"] = model_identity(model_path)
        payload["pipeline_version"] = PIPELINE_VERSION
        # Cấu hình sinh THỰC TẾ (chung + override của hệ thống), GỒM batch_size: đối chứng trên 225 văn bản dài cho
        # thấy đổi batch làm đổi output ở ~26% văn bản, nên batch khác nhau không được dùng chung cache.
        payload["generation"] = effective_generation(cfg, sys_cfg)
    if sys_cfg["type"] == "pipeline":
        from vnsum.config import load_config

        icfg = _inference_config(sys_cfg)
        payload["model_path"] = model_identity(model_path)
        payload["pipeline_version"] = PIPELINE_VERSION
        # Toàn bộ config suy luận + config dữ liệu (tiền xử lý, tách câu, lọc câu) đã resolve: đổi gì cũng sinh lại.
        payload["inference"] = {k: v for k, v in icfg.items() if k not in ("api", "benchmark")}
        payload["data"] = load_config(icfg["data_config"])
    if sys_cfg["type"] == "llm":
        client = dict(cfg["llm"][sys_cfg["client"]])
        client.pop("concurrency", None)
        payload["llm"] = client
    return fingerprint(payload)


def sample_input_hashes(df, sys_cfg: dict[str, Any]) -> list[str]:
    """Hash input của từng mẫu: cột input của hệ thống, cộng document gốc khi chạy hierarchical."""
    texts = df[sys_cfg["input_column"]].fillna("").astype(str)
    if sys_cfg.get("long_strategy") == "hierarchical":
        texts = texts + "\x00" + df["document"].fillna("").astype(str)
    return [input_hash(t) for t in texts]


def estimate_llm_requests(missing: int, texts: Sequence[str], max_chars: int) -> str:
    chars = sum(min(len(t), int(max_chars)) for t in texts)
    return f"{missing} request, ~{chars:,} ký tự input (chưa gồm prompt)"


def run_system(
    name: str,
    sys_cfg: dict[str, Any],
    df,
    cfg: dict[str, Any],
    *,
    results_dir: Path,
    gate: ApiGate,
    model_path: str | None = None,
) -> dict[str, str] | None:
    """Trả ``{id: prediction}`` cho mọi mẫu của ``df``; ``None`` nếu hệ thống bị bỏ qua (thiếu --allow_api)."""
    ids = [str(i) for i in df["id"]]
    is_llm = sys_cfg["type"] == "llm"
    cache = PredictionCache(results_dir / "predictions", name, system_fingerprint(name, sys_cfg, cfg, model_path),
                            accept_legacy=is_llm)
    hashes = dict(zip(ids, sample_input_hashes(df, sys_cfg)))
    missing = set(cache.missing(ids, [hashes[i] for i in ids]))
    if cache.legacy_used:
        logger.warning("%s: dùng lại %d dự đoán API từ cache cũ chưa có hash input (không kiểm tra được input có "
                       "đổi hay không; xóa %s nếu dữ liệu đã thay đổi).", name, cache.legacy_used, cache.path)
    todo = df[df["id"].astype(str).isin(missing)]
    if len(todo):
        texts = todo[sys_cfg["input_column"]].fillna("").tolist()
        kind = sys_cfg["type"]
        if kind == "lead":
            preds = [lead_n(t, int(sys_cfg.get("num_sentences", 3))) for t in texts]
        elif kind == "seq2seq" and sys_cfg.get("long_strategy") == "hierarchical":
            preds = run_hierarchical_seq2seq(todo, str(model_path), sys_cfg, cfg)
        elif kind == "pipeline":
            preds = generate_pipeline(texts, str(model_path), sys_cfg)
        elif kind == "seq2seq":
            preds = generate_seq2seq(texts, str(model_path), sys_cfg, cfg["generation"], cfg.get("hf", {}).get("token_env"))
        elif kind == "llm":
            client_cfg = cfg["llm"][sys_cfg["client"]]
            max_chars = int(sys_cfg.get("max_input_chars", 30000))
            gate.plan(name, len(todo))
            estimate = estimate_llm_requests(len(todo), texts, max_chars)
            if not gate.allowed:
                print(f"[{name}] BỎ QUA — cần --allow_api. Ước tính: {estimate} tới {client_cfg['provider']}/{client_cfg['model']}.")
                return None
            print(f"[{name}] Gọi API: {estimate} tới {client_cfg['provider']}/{client_cfg['model']}")
            client = LLMClient(client_cfg, label=name, gate=gate)
            generate_llm(todo["id"].astype(str).tolist(), texts, client, client_cfg["prompt"], max_chars, cache, hashes)
            preds = None
        else:  # pragma: no cover - đã kiểm tra trong config
            raise ValueError(kind)
        if preds is not None:
            cache.add(dict(zip(todo["id"].astype(str), preds)), hashes)
        logger.info("%s: sinh mới %d, dùng cache %d", name, len(todo), len(ids) - len(todo))
    else:
        logger.info("%s: dùng toàn bộ %d dự đoán từ cache", name, len(ids))
    return {i: cache.data[i] for i in ids}
