"""Tải VietNews và WikiLingua (tiếng Việt) từ Hugging Face Hub.

Cả hai đều được chuẩn hóa về cùng schema ``{id, document, summary}`` và trả
về ``dict[split, list[record]]`` với các split ``train``/``validation``/``test``.

Dataset ID đã được xác minh trên Hub ngày 2026-10-05 (xem README):

* VietNews: ``nam194/vietnews``. Có sẵn split train/validation/test; cột
  ``guid, title, abstract, article``; văn bản đã tách từ bằng ``_``.
* WikiLingua (vi): ``esdurmus/wiki_lingua``, config ``vietnamese``. Chỉ có
  split train; mỗi dòng là một bài wikiHow gồm nhiều section, mỗi section là một
  cặp (document, summary). Val/test được tự chia theo hash của ``url``.

Loader này chỉ đọc và đổi tên trường; mọi bước làm sạch nằm ở ``preprocess``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from itertools import islice
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
Record = dict[str, Any]


def _hf_load(ds_cfg: dict[str, Any], split: str, *, streaming: bool, cache_dir: str | None, token_env: str | None):
    from datasets import load_dataset  # import trễ: unit test không cần `datasets`

    token = os.environ.get(token_env) if token_env else None
    return load_dataset(
        ds_cfg["hf_id"],
        name=ds_cfg.get("hf_config"),
        split=split,
        revision=ds_cfg.get("revision"),
        streaming=streaming,
        cache_dir=cache_dir,
        token=token or None,
    )


def _take(ds: Iterable[Record], limit: int | None) -> Iterator[Record]:
    return iter(ds) if limit is None else islice(iter(ds), limit)


# --------------------------------------------------------------------------- #
# VietNews
# --------------------------------------------------------------------------- #


def standardize_vietnews(row: Record, split: str, fields: dict[str, str], name: str) -> Record:
    return {
        "id": f"{name}-{split}-{row[fields['id']]}",
        "document": row[fields["document"]] or "",
        "summary": row[fields["summary"]] or "",
    }


def load_vietnews(
    name: str,
    ds_cfg: dict[str, Any],
    *,
    limit_per_split: int | None,
    streaming: bool,
    cache_dir: str | None,
    token_env: str | None,
) -> dict[str, list[Record]]:
    out: dict[str, list[Record]] = {}
    for target, source in ds_cfg["splits"].items():
        ds = _hf_load(ds_cfg, source, streaming=streaming, cache_dir=cache_dir, token_env=token_env)
        out[target] = [standardize_vietnews(r, target, ds_cfg["fields"], name) for r in _take(ds, limit_per_split)]
        logger.info("%s/%s: đọc %d mẫu từ split nguồn %r", name, target, len(out[target]), source)
    return out


# --------------------------------------------------------------------------- #
# WikiLingua (vi)
# --------------------------------------------------------------------------- #


def _sections(article: Any) -> list[dict[str, Any]]:
    """``article`` có thể là list[dict] (JSON) hoặc dict[list] (``datasets.Sequence``)."""
    if article is None:
        return []
    if isinstance(article, dict):
        keys = list(article.keys())
        n = len(article[keys[0]]) if keys else 0
        return [{k: article[k][i] for k in keys} for i in range(n)]
    return list(article)


def assign_split(key: str, seed: int, ratios: dict[str, float]) -> str:
    """Chia split xác định theo hash(seed, key), không phụ thuộc thứ tự đọc."""
    h = int(hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest(), 16) / float(1 << 128)
    acc = 0.0
    names = [s for s in SPLITS if s in ratios] + [s for s in ratios if s not in SPLITS]
    for split in names:
        acc += float(ratios[split])
        if h < acc:
            return split
    return names[-1]


def standardize_wikilingua(row: Record, name: str, split_key: str) -> list[Record]:
    key = row[split_key]
    stem = hashlib.md5(str(key).encode("utf-8")).hexdigest()[:12]
    return [
        {"id": f"{name}-{stem}-{i}", "document": sec.get("document") or "", "summary": sec.get("summary") or ""}
        for i, sec in enumerate(_sections(row.get("article")))
    ]


def load_wikilingua(
    name: str,
    ds_cfg: dict[str, Any],
    *,
    seed: int,
    limit_per_split: int | None,
    streaming: bool,
    max_scan_records: int | None,
    cache_dir: str | None,
    token_env: str | None,
) -> dict[str, list[Record]]:
    ratios = ds_cfg["split_ratios"]
    split_key = ds_cfg.get("split_key", "url")
    out: dict[str, list[Record]] = {s: [] for s in ratios}
    ds = _hf_load(ds_cfg, ds_cfg["source_split"], streaming=streaming, cache_dir=cache_dir, token_env=token_env)

    scanned = 0
    for row in ds:
        scanned += 1
        split = assign_split(str(row[split_key]), seed, ratios)
        if limit_per_split is None or len(out[split]) < limit_per_split:
            recs = standardize_wikilingua(row, name, split_key)
            if limit_per_split is not None:
                recs = recs[: limit_per_split - len(out[split])]
            out[split].extend(recs)
        if limit_per_split is not None:
            if all(len(v) >= limit_per_split for v in out.values()):
                break
            if max_scan_records is not None and scanned >= max_scan_records:
                logger.warning("%s: dừng sau %d bài (max_scan_records_when_limited)", name, scanned)
                break
    for split, recs in out.items():
        logger.info("%s/%s: %d mẫu (đã quét %d bài)", name, split, len(recs), scanned)
    return out


# --------------------------------------------------------------------------- #
# Điểm vào chung
# --------------------------------------------------------------------------- #

_LOADERS = {"vietnews": "vietnews", "wikilingua_vi": "wikilingua"}


def load_splits(name: str, cfg: dict[str, Any], limit_per_split: int | None = None) -> dict[str, list[Record]]:
    """Tải dataset ``name`` (khóa trong ``cfg['datasets']``) về schema chung."""
    ds_cfg = cfg["datasets"][name]
    kind = _LOADERS.get(name, ds_cfg.get("loader"))
    streaming = bool(limit_per_split) and bool(cfg["loading"].get("streaming_when_limited", True))
    common = {
        "limit_per_split": limit_per_split,
        "streaming": streaming,
        "cache_dir": cfg["paths"].get("hf_cache_dir"),
        "token_env": cfg["hf"].get("token_env"),
    }
    if kind == "vietnews":
        return load_vietnews(name, ds_cfg, **common)
    if kind == "wikilingua":
        return load_wikilingua(
            name,
            ds_cfg,
            seed=int(cfg["seed"]),
            max_scan_records=cfg["loading"].get("max_scan_records_when_limited"),
            **common,
        )
    raise ValueError(f"Không có loader cho dataset {name!r}")
