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


class PredictionCache:
    """``<dir>/<system>.jsonl``: mỗi dòng ``{id, prediction, fp}``. Chỉ dùng lại dòng có cùng fingerprint."""

    def __init__(self, directory: str | Path, system: str, fp: str) -> None:
        self.path = Path(directory) / f"{system}.jsonl"
        self.fp = fp
        self._lock = threading.Lock()
        self.data: dict[str, str] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("fp") == fp:
                        self.data[str(row["id"])] = row["prediction"]

    def missing(self, ids: Sequence[str]) -> list[str]:
        return [i for i in ids if str(i) not in self.data]

    def add(self, items: dict[str, str]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for i, pred in items.items():
                    self.data[str(i)] = pred
                    f.write(json.dumps({"id": str(i), "prediction": pred, "fp": self.fp}, ensure_ascii=False) + "\n")


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


def generate_seq2seq(texts: Sequence[str], model_path: str, sys_cfg: dict[str, Any], gen: dict[str, Any], token_env: str | None) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM

    from vnsum.data.tokenization import load_tokenizer

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(gen.get("dtype", "auto"))
    token = os.environ.get(token_env) if token_env else None
    logger.info("Load %s (%s, %s)", model_path, device, dtype)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=token or None, dtype=dtype).to(device).eval()
    tok = load_tokenizer(model_path, token_env)
    prefix = sys_cfg.get("source_prefix") or ""
    bs = int(gen["batch_size"])
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # gom input cùng độ dài -> ít padding
    preds: list[str] = [""] * len(texts)
    for start in tqdm(range(0, len(order), bs), desc=f"generate {Path(str(model_path)).name}"):
        idx = order[start : start + bs]
        enc = tok(
            [prefix + texts[i] for i in idx],
            max_length=int(sys_cfg["max_source_length"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=int(gen["num_beams"]),
                max_new_tokens=int(gen["max_new_tokens"]),
                no_repeat_ngram_size=int(gen["no_repeat_ngram_size"]),
                early_stopping=bool(gen.get("early_stopping", True)),
            )
        for i, text in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
            preds[i] = text.strip()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


# --------------------------------------------------------------------------- #
# LLM zero-shot
# --------------------------------------------------------------------------- #


def build_llm_prompt(template: str, document: str, max_chars: int) -> str:
    return template.replace("{document}", document[: int(max_chars)])


def generate_llm(
    ids: Sequence[str], texts: Sequence[str], client: LLMClient, template: str, max_chars: int, cache: PredictionCache
) -> None:
    """Gọi API cho từng văn bản; ghi cache ngay sau mỗi request (dừng giữa chừng không mất phần đã trả phí)."""

    def _one(pair: tuple[str, str]) -> None:
        i, doc = pair
        text, _ = client.complete(build_llm_prompt(template, doc, max_chars))
        cache.add({i: text})

    client.map(list(zip(ids, texts)), _one)


# --------------------------------------------------------------------------- #
# Điểm vào: chạy (hoặc lấy cache) cho một hệ thống
# --------------------------------------------------------------------------- #


def system_fingerprint(name: str, sys_cfg: dict[str, Any], cfg: dict[str, Any], model_path: str | None) -> str:
    # display_name/description chỉ ảnh hưởng report -> không làm mất cache dự đoán khi đổi tên.
    shown = {k: v for k, v in sys_cfg.items() if k not in ("display_name", "description")}
    payload: dict[str, Any] = {"name": name, "system": shown, "model_path": model_path}
    if sys_cfg["type"] == "seq2seq":
        payload["generation"] = {k: v for k, v in cfg["generation"].items() if k != "batch_size"}
    if sys_cfg["type"] == "llm":
        client = dict(cfg["llm"][sys_cfg["client"]])
        client.pop("concurrency", None)
        payload["llm"] = client
    return fingerprint(payload)


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
    cache = PredictionCache(results_dir / "predictions", name, system_fingerprint(name, sys_cfg, cfg, model_path))
    missing = set(cache.missing(ids))
    todo = df[df["id"].astype(str).isin(missing)]
    if len(todo):
        texts = todo[sys_cfg["input_column"]].fillna("").tolist()
        kind = sys_cfg["type"]
        if kind == "lead":
            preds = [lead_n(t, int(sys_cfg.get("num_sentences", 3))) for t in texts]
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
            generate_llm(todo["id"].astype(str).tolist(), texts, client, client_cfg["prompt"], max_chars, cache)
            preds = None
        else:  # pragma: no cover - đã kiểm tra trong config
            raise ValueError(kind)
        if preds is not None:
            cache.add(dict(zip(todo["id"].astype(str), preds)))
        logger.info("%s: sinh mới %d, dùng cache %d", name, len(todo), len(ids) - len(todo))
    else:
        logger.info("%s: dùng toàn bộ %d dự đoán từ cache", name, len(ids))
    return {i: cache.data[i] for i in ids}
