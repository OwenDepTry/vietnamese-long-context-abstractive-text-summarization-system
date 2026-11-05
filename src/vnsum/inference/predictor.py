"""Summarizer: một interface chung cho hai backend (PyTorch, ONNX Runtime).

    from vnsum.inference.predictor import Summarizer
    s = Summarizer.from_config(cfg, model_path="C:/vnsum-runs/vit5-vnsum-merged")      # backend theo cfg
    r = s.summarize(van_ban, length="short")
    print(r.summary, r.strategy, r.timings_ms)

Tiền xử lý dùng lại code phase 1 (chuẩn hóa, tách câu, extractive_filter, chunk_document), nên
input của model giống hệt lúc train. Văn bản dài được xử lý theo ``long_document.strategy``:

* ``direct``: vừa ``max_input_tokens`` -> đưa thẳng vào model.
* ``extractive``: chọn câu bằng BM25 tới khi đầy ngân sách token (giống input lúc train / phase 3).
* ``hierarchical``: chia chunk theo câu (có overlap), tóm tắt từng chunk, rồi bước reduce
  (``long_document.reduce``, xem ``vnsum.inference.reduce``):
  - ``generate``: ghép các bản tóm tắt rồi tóm tắt lần cuối; nếu bản ghép vẫn quá dài thì lặp lại
    (tối đa ``max_rounds`` vòng).
  - ``concat``: ghép các bản tóm tắt, bỏ câu trùng, chọn câu nếu dài hơn ``max_new_tokens`` của mức
    độ dài được yêu cầu; không sinh thêm lần nào.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from vnsum.data.long_context import chunk_document, extractive_filter
from vnsum.data.preprocess import normalize_text, split_sentences
from vnsum.data.tokenization import TokenCounter
from vnsum.inference.config import generation_kwargs, length_preset, onnx_dirs, resolve_model_path
from vnsum.inference.reduce import reduce_concat

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- backends
@dataclass
class GenerationOutput:
    texts: list[str]
    new_tokens: list[int]          # số token sinh ra (gồm </s>, không gồm padding / decoder start)


class Backend(Protocol):
    name: str
    device: str
    dtype: str
    tokenizer: Any

    def generate(self, texts: Sequence[str], gen_kwargs: dict[str, Any]) -> GenerationOutput: ...


def _count_new_tokens(sequences, pad_id: int) -> list[int]:
    # T5: decoder_start_token_id = pad_token_id, nên đếm token khác pad = số token sinh ra (gồm </s>).
    return [int((seq != pad_id).sum()) for seq in sequences]


def _generate(model, tokenizer, texts: Sequence[str], gen_kwargs: dict[str, Any], max_input_tokens: int,
              device) -> GenerationOutput:
    import torch

    enc = tokenizer(list(texts), max_length=int(max_input_tokens), truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    out = out.cpu()
    texts_out = [t.strip() for t in tokenizer.batch_decode(out, skip_special_tokens=True)]
    return GenerationOutput(texts_out, _count_new_tokens(out, tokenizer.pad_token_id))


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda nhưng torch không thấy GPU (cài torch bản CUDA?)")
    return name


class PyTorchBackend:
    """Model PyTorch đã merge (``AutoModelForSeq2SeqLM``)."""

    name = "pytorch"

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "auto", max_input_tokens: int = 1024) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM

        from vnsum.inference.tokenizer import load_tokenizer_compat

        self.device = resolve_device(device)
        if self.device == "cpu" or dtype == "fp32":
            torch_dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif dtype == "bf16":
            raise RuntimeError(f"GPU {torch.cuda.get_device_name(0)} không hỗ trợ bf16; đặt pytorch.dtype: fp32")
        else:
            torch_dtype = torch.float32
        self.dtype = {torch.float32: "fp32", torch.bfloat16: "bf16"}[torch_dtype]
        # Nạp rồi .to(dtype): tên tham số dtype khác nhau giữa transformers 4 (torch_dtype) và 5 (dtype).
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device=self.device, dtype=torch_dtype).eval()
        self.tokenizer = load_tokenizer_compat(model_path)
        self.max_input_tokens = int(max_input_tokens)
        self.model_dir = Path(model_path)

    def generate(self, texts: Sequence[str], gen_kwargs: dict[str, Any]) -> GenerationOutput:
        return _generate(self.model, self.tokenizer, texts, gen_kwargs, self.max_input_tokens, self.device)


def _cuda_provider_ready() -> None:
    """ORT-GPU trên Windows cần DLL CUDA/cuDNN: nạp torch (bản CUDA) trước, rồi preload nếu ORT hỗ trợ."""
    import torch  # noqa: F401  (nạp DLL CUDA của torch vào process)
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime không có CUDAExecutionProvider. Cài onnxruntime-gpu thay cho onnxruntime "
                           "(pip uninstall -y onnxruntime && pip install onnxruntime-gpu).")
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as err:  # pragma: no cover - phụ thuộc máy
            logger.warning("onnxruntime.preload_dlls() lỗi (%s); thử tiếp với DLL torch đã nạp.", err)


def _session_providers(model) -> list[str]:
    for attr in ("providers",):
        v = getattr(model, attr, None)
        if v:
            return list(v)
    sess = getattr(getattr(model, "encoder", None), "session", None)
    return list(sess.get_providers()) if sess is not None else []


class OnnxBackend:
    """``optimum.onnxruntime.ORTModelForSeq2SeqLM`` (encoder + decoder + decoder_with_past, có KV cache)."""

    name = "onnx"

    def __init__(self, onnx_dir: str | Path, variant: str, device: str = "auto", intra_op_num_threads: int | None = None,
                 max_input_tokens: int = 1024, *, allow_spinning: bool | None = None,
                 use_io_binding: bool | None = None) -> None:
        import onnxruntime as ort

        onnx_dir = Path(onnx_dir)
        if not any(onnx_dir.glob("*.onnx")):
            raise FileNotFoundError(f"Không có file .onnx trong {onnx_dir}. Chạy scripts/export_onnx.py trước.")
        if device == "auto":
            device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        if variant == "fp16" and device != "cuda":
            raise RuntimeError("Bản ONNX fp16 chỉ dùng trên GPU (onnx.device: cuda)")
        if device == "cuda":
            _cuda_provider_ready()
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        from vnsum.inference.tokenizer import load_tokenizer_compat

        so = ort.SessionOptions()
        if intra_op_num_threads:
            so.intra_op_num_threads = int(intra_op_num_threads)
        if allow_spinning is not None:
            # 3 session (encoder, decoder, decoder_with_past) có 3 thread pool riêng; tắt spin-wait để
            # pool đang rảnh không chiếm CPU của pool đang chạy.
            so.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        kwargs: dict[str, Any] = {}
        if use_io_binding is not None:
            kwargs["use_io_binding"] = bool(use_io_binding)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir, provider=provider, session_options=so,
                                                          use_cache=True, use_merged=False, **kwargs)
        self.session_settings = {"intra_op_num_threads": so.intra_op_num_threads or "mặc định ORT",
                                 "allow_spinning": "mặc định ORT" if allow_spinning is None else allow_spinning,
                                 "use_io_binding": getattr(self.model, "use_io_binding", use_io_binding)}
        providers = _session_providers(self.model)
        if device == "cuda" and providers and providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX Runtime không dùng được GPU (providers thực tế: {providers}). "
                               "Kiểm tra bản CUDA/cuDNN tương thích với onnxruntime-gpu.")
        self.providers = providers
        self.device, self.dtype = device, variant
        self.tokenizer = load_tokenizer_compat(onnx_dir)
        self.max_input_tokens = int(max_input_tokens)
        self.model_dir = onnx_dir
        self.intra_op_num_threads = so.intra_op_num_threads

    def generate(self, texts: Sequence[str], gen_kwargs: dict[str, Any]) -> GenerationOutput:
        return _generate(self.model, self.tokenizer, texts, gen_kwargs, self.max_input_tokens,
                         "cuda" if self.device == "cuda" else "cpu")


def onnx_session_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    o = cfg["onnx"]
    return {"intra_op_num_threads": o.get("intra_op_num_threads"), "allow_spinning": o.get("allow_spinning"),
            "use_io_binding": o.get("use_io_binding")}


def build_backend(cfg: dict[str, Any], backend: str | None = None, *, model_path: str | None = None,
                  variant: str | None = None, device: str | None = None) -> Backend:
    backend = backend or cfg["backend"]
    model_path = resolve_model_path(cfg, model_path)
    if backend == "pytorch":
        return PyTorchBackend(model_path, device or cfg["pytorch"].get("device", "auto"),
                              cfg["pytorch"].get("dtype", "auto"), int(cfg["max_input_tokens"]))
    if backend == "onnx":
        variant = variant or cfg["onnx"]["variant"]
        return OnnxBackend(onnx_dirs(cfg, model_path)[variant], variant, device or cfg["onnx"].get("device", "auto"),
                           **onnx_session_kwargs(cfg), max_input_tokens=int(cfg["max_input_tokens"]))
    raise ValueError(f"backend không hợp lệ: {backend!r}")


# --------------------------------------------------------------------------- summarizer
@dataclass
class SummaryResult:
    summary: str
    length: str
    strategy: str                       # direct | extractive | hierarchical
    input_tokens: int                   # token của văn bản sau chuẩn hóa (trước khi cắt/lọc)
    model_input_tokens: int             # token thực sự đưa vào lần sinh cuối
    new_tokens: int                     # token sinh ra ở lần sinh cuối
    generate_calls: int                 # số lần gọi model (các chunk được sinh theo batch)
    timings_ms: dict[str, float] = field(default_factory=dict)
    chunk_summaries: list[str] = field(default_factory=list)
    model_input: str = ""
    chunk_cache_hit: bool = False       # bản tóm tắt chunk lấy lại từ lần gọi trước (cùng văn bản)
    unsupported: list[str] = field(default_factory=list)   # số / ngày / tên riêng không có trong văn bản gốc


class Summarizer:
    def __init__(self, backend: Backend, cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
        self.backend = backend
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.counter = TokenCounter(backend.tokenizer, int(data_cfg["tokenizer"].get("reserve_special_tokens", 1)))
        sc = data_cfg.get("sentence_split", {}) or {}
        backend_name, min_chars = sc.get("backend", "rule"), int(sc.get("min_chars", 2))
        self.splitter = lambda t: split_sentences(t, backend=backend_name, min_chars=min_chars)
        # Bản tóm tắt chunk của các văn bản gần nhất (LRU). UI gọi short / medium / long lần lượt cho CÙNG
        # văn bản; với reduce concat cả ba dùng cùng bản tóm tắt chunk, chỉ khác ngân sách chọn câu cuối.
        self._chunk_cache: OrderedDict[str, list[str]] = OrderedDict()
        self._chunk_lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], backend: str | None = None, **kwargs) -> "Summarizer":
        from vnsum.config import load_config

        data_cfg = load_config(cfg["data_config"])
        return cls(build_backend(cfg, backend, **kwargs), cfg, data_cfg)

    # ----- các bước
    def normalize(self, text: str) -> str:
        return normalize_text(text, self.data_cfg["preprocess"])

    def _budget(self) -> int:
        return self.counter.budget(int(self.cfg["max_input_tokens"]))

    def choose_strategy(self, n_tokens: int, strategy: str) -> str:
        if n_tokens <= self._budget():
            return "direct"
        if strategy == "auto":
            limit = int(self.cfg["long_document"]["hierarchical_above_tokens"])
            return "extractive" if n_tokens <= limit else "hierarchical"
        return strategy

    def extract(self, text: str, max_input_tokens: int | None = None) -> str:
        ex = self.data_cfg["long_context"]["extractive_filter"]
        ranker = ex["ranker"]
        return extractive_filter(
            text, self.counter, max_input_tokens=int(max_input_tokens or self.cfg["max_input_tokens"]),
            ranker=ranker, selection=ex["selection"], keep_ratio=float(ex["keep_ratio"]),
            min_sentences=int(ex["min_sentences"]), only_if_exceeds=bool(ex["only_if_exceeds"]),
            ranker_params=dict(ex.get(ranker, {})), splitter=self.splitter,
        ).text

    def _gen(self, texts: list[str], length: str) -> GenerationOutput:
        return self.backend.generate(texts, generation_kwargs(self.cfg, length))

    def _chunk_batch_size(self) -> int:
        size = int(self.cfg["long_document"].get("chunk_batch_size", 1))
        limit = getattr(self.backend, "max_batch_size", None)   # vd. streaming chỉ hỗ trợ batch 1
        return max(1, min(size, int(limit)) if limit else size)

    def _gen_chunks(self, texts: list[str], length: str) -> list[str]:
        """Tóm tắt nhiều chunk, theo batch ``long_document.chunk_batch_size`` (thay vì từng chunk một)."""
        n = self._chunk_batch_size()
        out: list[str] = []
        for i in range(0, len(texts), n):
            out.extend(self._gen(texts[i:i + n], length).texts)
        return out

    def _n_batches(self, n_texts: int) -> int:
        """Số lần gọi model để tóm tắt ``n_texts`` chunk (= ``generate_calls``)."""
        return -(-n_texts // self._chunk_batch_size())

    def _chunk_key(self, doc: str) -> str:
        ld = self.cfg["long_document"]
        sig = json.dumps([ld["chunk_max_tokens"], ld["overlap_tokens"], ld["chunk_length"],
                          generation_kwargs(self.cfg, ld["chunk_length"])], sort_keys=True, default=str)
        return hashlib.sha1((sig + "\x00" + doc).encode("utf-8")).hexdigest()

    def _cached_chunk_summaries(self, doc: str, chunks: list) -> tuple[list[str], bool]:
        size = int(self.cfg["long_document"].get("chunk_cache_size", 0))
        key = self._chunk_key(doc) if size > 0 else ""
        if size > 0:
            with self._chunk_lock:
                if key in self._chunk_cache:
                    self._chunk_cache.move_to_end(key)
                    return list(self._chunk_cache[key]), True
        parts = self._gen_chunks([c.text for c in chunks], self.cfg["long_document"]["chunk_length"])
        if size > 0:
            with self._chunk_lock:
                self._chunk_cache[key] = list(parts)
                self._chunk_cache.move_to_end(key)
                while len(self._chunk_cache) > size:
                    self._chunk_cache.popitem(last=False)
        return parts, False

    def summarize(self, text: str, length: str | None = None, strategy: str | None = None) -> SummaryResult:
        result = self._summarize(text, length, strategy)
        if self.cfg.get("factcheck_warnings", True):
            # Cảnh báo, không sửa bản tóm tắt: số / ngày / tên riêng không tìm thấy trong văn bản gốc.
            from vnsum.factcheck import check_summary

            fc = check_summary(result.summary, self.normalize(text), self.splitter)
            result.unsupported = [x for u in fc.unsupported for x in u]
        return result

    def _summarize(self, text: str, length: str | None = None, strategy: str | None = None) -> SummaryResult:
        length = length or self.cfg["default_length"]
        generation_kwargs(self.cfg, length)  # kiểm tra length sớm
        strategy = strategy or self.cfg["long_document"]["strategy"]
        t0 = time.perf_counter()
        doc = self.normalize(text)
        if not doc.strip():
            raise ValueError("Văn bản rỗng sau khi chuẩn hóa")
        n_tok = self.counter.count(doc)
        used = self.choose_strategy(n_tok, strategy)
        timings: dict[str, float] = {}
        chunk_summaries: list[str] = []
        calls = 0

        if used == "hierarchical" and self.cfg["long_document"].get("reduce", "generate") == "concat":
            return self._hierarchical_concat(doc, n_tok, length, t0)
        if used == "hierarchical":
            ld = self.cfg["long_document"]
            current = doc
            for _ in range(int(ld["max_rounds"])):
                chunks = chunk_document(current, self.counter, chunk_max_tokens=int(ld["chunk_max_tokens"]),
                                        overlap_tokens=int(ld["overlap_tokens"]), splitter=self.splitter)
                t = time.perf_counter()
                parts = self._gen_chunks([c.text for c in chunks], ld["chunk_length"])
                timings["generate_chunks_ms"] = timings.get("generate_chunks_ms", 0.0) + (time.perf_counter() - t) * 1000
                calls += self._n_batches(len(chunks))
                if not chunk_summaries:  # giữ bản tóm tắt chunk của vòng đầu để hiển thị / debug
                    chunk_summaries = parts
                current = " ".join(p for p in parts if p)
                if self.counter.count(current) <= self._budget():
                    break
            model_input = current if self.counter.count(current) <= self._budget() else self.extract(current)
        elif used == "extractive":
            model_input = self.extract(doc)
        else:
            model_input = doc
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000 - timings.get("generate_chunks_ms", 0.0)

        t = time.perf_counter()
        out = self._gen([model_input], length)
        timings["generate_ms"] = (time.perf_counter() - t) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return SummaryResult(
            summary=out.texts[0], length=length, strategy=used, input_tokens=n_tok,
            model_input_tokens=self.counter.count(model_input) + self.counter.reserve_special_tokens,
            new_tokens=out.new_tokens[0], generate_calls=calls + 1, timings_ms=timings,
            chunk_summaries=chunk_summaries, model_input=model_input,
        )

    def _hierarchical_concat(self, doc: str, n_tok: int, length: str, t0: float) -> SummaryResult:
        """Tóm tắt từng chunk một lần, rồi ghép + bỏ câu trùng + chọn câu (không sinh lại)."""
        ld = self.cfg["long_document"]
        chunks = chunk_document(doc, self.counter, chunk_max_tokens=int(ld["chunk_max_tokens"]),
                                overlap_tokens=int(ld["overlap_tokens"]), splitter=self.splitter)
        t_pre = (time.perf_counter() - t0) * 1000
        t = time.perf_counter()
        parts, hit = self._cached_chunk_summaries(doc, chunks)
        gen_ms = (time.perf_counter() - t) * 1000
        max_tokens = int(length_preset(self.cfg, length)["max_new_tokens"])
        reserve = self.counter.reserve_special_tokens
        summary = reduce_concat(
            parts, splitter=self.splitter, count_tokens=self.counter.count, max_tokens=max_tokens,
            select=lambda text, n: self.extract(text, max_input_tokens=n + reserve),
            dedupe_threshold=float(ld.get("dedupe_threshold", 0.6)),
            source=doc, factcheck=ld.get("factcheck", "off"),
        )
        joined = " ".join(p for p in parts if p)
        timings = {"preprocess_ms": t_pre, "generate_chunks_ms": gen_ms, "generate_ms": 0.0,
                   "total_ms": (time.perf_counter() - t0) * 1000}
        return SummaryResult(
            summary=summary, length=length, strategy="hierarchical", input_tokens=n_tok,
            model_input_tokens=max((self.counter.count(c.text) + reserve for c in chunks), default=0),
            new_tokens=self.counter.count(summary), generate_calls=0 if hit else self._n_batches(len(chunks)), timings_ms=timings,
            chunk_summaries=parts, model_input=joined, chunk_cache_hit=hit,
        )
