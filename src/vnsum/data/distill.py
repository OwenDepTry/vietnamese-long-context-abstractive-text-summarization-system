"""Bước mở rộng B — dữ liệu distillation: LLM viết bản tóm tắt đầy đủ ý cho một phần tập TRAIN.

LLM nhận đúng ``input_text`` mà ViT5 sẽ thấy lúc train (đã lọc BM25 ≤ 1024 token), không phải document
gốc: nếu LLM tóm tắt cả phần ViT5 không được thấy, model học cách "bịa" thông tin không có trong input.

Chỉ dùng pandas + thư viện chuẩn; gọi API qua ``LLMClient``/``ApiGate`` của phase 3 (cần --allow_api).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_PREFIX_RE = re.compile(r"^\s*(?:\*\*)?\s*(?:tóm tắt|bản tóm tắt|summary)\s*(?:\*\*)?\s*[:：\-–]\s*(?:\*\*)?\s*", re.IGNORECASE)
_QUOTES = "\"'“”‘’«»"


def sample_train(processed_dir: str | Path, counts: dict[str, int], seed: int) -> pd.DataFrame:
    """Lấy mẫu cố định (seed) ``counts[dataset]`` bài từ split train -> id, dataset, input_text, summary, document_tokens."""
    frames = []
    for name, n in counts.items():
        path = Path(processed_dir) / name / "train.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Không thấy {path}. Chạy phase 1 trước (scripts/prepare_data.py).")
        df = pd.read_parquet(path, columns=["id", "input_text", "summary", "document_tokens"])
        df = df[df["input_text"].fillna("").str.strip().str.len() > 0]
        df = df.sample(n=min(int(n), len(df)), random_state=int(seed))
        df.insert(1, "dataset", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["id"] = out["id"].astype(str)
    return out


def build_prompt(template: str, text: str, max_chars: int) -> str:
    return template.replace("{document}", text[: int(max_chars)])


def clean_summary(text: str) -> str:
    """Bỏ tiền tố kiểu "Tóm tắt:", dấu ngoặc kép bao ngoài, gạch đầu dòng; gộp về một đoạn."""
    t = (text or "").strip()
    t = _PREFIX_RE.sub("", t)
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", ln).strip() for ln in t.splitlines()]
    t = " ".join(ln for ln in lines if ln)
    if len(t) >= 2 and t[0] in _QUOTES and t[-1] in _QUOTES:
        t = t[1:-1].strip()
    return re.sub(r"\s+", " ", t)


def filter_summaries(
    df: pd.DataFrame, count_tokens: Callable[[list[str]], list[int]], *, max_target_tokens: int, min_syllables: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Giữ bản tóm tắt dùng được để train. ``df`` có cột ``distill_summary``.

    Bỏ: rỗng / từ chối; quá ngắn; dài hơn ``max_target_tokens`` token ViT5 (sẽ bị cắt cụt lúc train ->
    model học câu bỏ dở). Trả (df giữ lại, số mẫu bỏ theo lý do).
    """
    s = df["distill_summary"].fillna("").map(clean_summary)
    syl = s.str.split().str.len().fillna(0).astype(int)
    tok = pd.Series(count_tokens(s.tolist()), index=df.index)
    refusal = s.str.contains(r"^(?:xin lỗi|tôi không thể|i'm sorry|i cannot)", case=False, regex=True)
    reasons = pd.Series("", index=df.index)
    reasons[s.str.len() == 0] = "rỗng"
    reasons[(reasons == "") & refusal] = "từ chối"
    reasons[(reasons == "") & (syl < int(min_syllables))] = f"< {min_syllables} âm tiết"
    reasons[(reasons == "") & (tok > int(max_target_tokens))] = f"> {max_target_tokens} token"
    kept = df[reasons == ""].copy()
    kept["distill_summary"] = s[reasons == ""]
    kept["distill_tokens"] = tok[reasons == ""]
    dropped = reasons[reasons != ""].value_counts().to_dict()
    return kept, {str(k): int(v) for k, v in dropped.items()}


def estimate_cost(prompts: list[str], price: dict[str, Any], expected_output_tokens: int) -> dict[str, float]:
    """Ước tính THÔ (ký tự / chars_per_token), chỉ để quyết định có chạy hay không — không phải số đo."""
    chars = sum(len(p) for p in prompts)
    in_tok = chars / float(price["chars_per_token"])
    out_tok = len(prompts) * int(expected_output_tokens)
    usd = in_tok / 1e6 * float(price["input_per_mtok"]) + out_tok / 1e6 * float(price["output_per_mtok"])
    return {"requests": len(prompts), "chars": chars, "input_tokens": in_tok, "output_tokens": out_tok, "usd": usd}
