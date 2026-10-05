"""Faithfulness bằng LLM-as-Judge: rubric 1–5, temperature 0, output JSON, có cache.

Judge chỉ so bản tóm tắt với VĂN BẢN GỐC (không xem reference), không yêu cầu
chain-of-thought: chỉ trả JSON ``{score, unsupported_claims, rationale}``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Sequence

from vnsum.eval.llm_client import ApiGate, LLMClient

logger = logging.getLogger(__name__)

RUBRIC = """\
5 — Hoàn toàn trung thực: mọi thông tin trong bản tóm tắt đều có trong văn bản gốc hoặc suy ra trực tiếp, chắc chắn từ văn bản gốc.
4 — Gần như trung thực: có chi tiết nhỏ không được văn bản gốc nêu rõ (diễn đạt mạnh hơn, khái quát hơi quá) nhưng không mâu thuẫn và không làm sai ý chính.
3 — Có thông tin không kiểm chứng được: ít nhất một thông tin đáng kể không có trong văn bản gốc, nhưng không mâu thuẫn trực tiếp với văn bản gốc.
2 — Có sai lệch: ít nhất một chi tiết mâu thuẫn với văn bản gốc (số liệu, tên, thời gian, địa điểm, quan hệ) nhưng ý chính vẫn đúng.
1 — Sai lệch sự thật nghiêm trọng: mâu thuẫn với nội dung chính, gán sai chủ thể/hành động, hoặc bịa ra sự kiện không có trong văn bản gốc."""

SYSTEM_PROMPT = (
    "Bạn là người chấm điểm độ trung thực (faithfulness) của bản tóm tắt tiếng Việt so với văn bản gốc. "
    "Chỉ đánh giá bản tóm tắt có được văn bản gốc hỗ trợ hay không; KHÔNG chấm văn phong, độ đầy đủ hay độ dài. "
    "Trả lời DUY NHẤT bằng một object JSON hợp lệ, không thêm chữ nào khác."
)

USER_TEMPLATE = """\
Thang điểm:
{rubric}

Văn bản gốc:
<document>
{document}
</document>

Bản tóm tắt cần chấm:
<summary>
{summary}
</summary>

Trả về JSON đúng dạng:
{{"score": <số nguyên 1-5>, "unsupported_claims": [<từng thông tin trong bản tóm tắt không được văn bản gốc hỗ trợ hoặc mâu thuẫn, trích ngắn gọn; [] nếu không có>], "rationale": "<tối đa 2 câu>"}}"""


class JudgeParseError(ValueError):
    pass


def build_judge_prompt(document: str, summary: str, max_chars: int) -> tuple[str, bool]:
    truncated = len(document) > int(max_chars)
    doc = document[: int(max_chars)]
    return USER_TEMPLATE.format(rubric=RUBRIC, document=doc, summary=summary), truncated


_SENT_END = re.compile(r"(?<=[.!?…])\s+")


def parse_judge_output(text: str) -> dict[str, Any]:
    """Lấy object JSON đầu tiên trong output, kiểm tra kiểu; rationale bị cắt còn ≤ 2 câu."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise JudgeParseError(f"Không tìm thấy JSON: {text[:200]!r}")
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as err:
        raise JudgeParseError(f"JSON lỗi ({err}): {text[:200]!r}") from None
    score = obj.get("score")
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise JudgeParseError(f"score không hợp lệ: {score!r}")
    claims = obj.get("unsupported_claims", [])
    if not isinstance(claims, list):
        raise JudgeParseError("unsupported_claims phải là list")
    rationale = str(obj.get("rationale", "")).strip()
    rationale = " ".join(_SENT_END.split(rationale)[:2])
    return {"score": score, "unsupported_claims": [str(c) for c in claims], "rationale": rationale}


_SCORE_RE = re.compile(r'^\s*(?:```(?:json)?\s*)?\{\s*"score"\s*:\s*([1-5])\s*[,}]')
_CLAIMS_RE = re.compile(r'"unsupported_claims"\s*:\s*\[(.*?)\]\s*,\s*"rationale"', re.S)
_RATIONALE_RE = re.compile(r'"rationale"\s*:\s*"(.*?)"?\s*\}?\s*(?:```)?\s*$', re.S)


def recover_judge_output(text: str) -> dict[str, Any] | None:
    """Phục hồi kết quả khi JSON không hợp lệ (vd. judge để dấu " không escape trong unsupported_claims,
    hoặc output bị cắt). Chỉ chấp nhận khi ``score`` là trường ĐẦU TIÊN và là số nguyên 1–5 — phần này
    không bị ảnh hưởng bởi lỗi ở phía sau. Không đoán điểm; trả None nếu không thấy score rõ ràng."""
    m = _SCORE_RE.match(text)
    if not m:
        return None
    c = _CLAIMS_RE.search(text)
    claims_raw = c.group(1).strip() if c else ""
    r = _RATIONALE_RE.search(text)
    rationale = " ".join(_SENT_END.split(r.group(1).strip())[:2]) if r else ""
    return {"score": int(m.group(1)), "unsupported_claims": [claims_raw] if claims_raw else [],
            "rationale": rationale, "recovered": True}


def resolve_cached(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Dòng cache bị lỗi parse nhưng có ``raw`` -> thử phục hồi (không gọi API, không sửa file cache)."""
    if not row or row.get("error") is None or not row.get("raw"):
        return row
    rec = recover_judge_output(row["raw"])
    if rec is None:
        return row
    return {**row, **rec, "error": None, "original_error": row["error"]}


def judge_sample_ids(ids: Sequence[str], n: int, seed: int) -> list[str]:
    """Mẫu ngẫu nhiên cố định theo seed, không phụ thuộc thứ tự đầu vào, và LỒNG NHAU:
    mẫu cỡ n luôn chứa mẫu cỡ m < n (tăng sample_size không làm phí kết quả judge đã cache)."""
    import random

    pool = sorted(str(i) for i in ids)
    random.Random(int(seed)).shuffle(pool)
    return sorted(pool[: min(int(n), len(pool))])


def cache_key(judge_model: str, prompt_version: str, system: str, sample_id: str, prediction: str) -> str:
    raw = json.dumps([judge_model, prompt_version, system, sample_id, prediction], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class JudgeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.data[row["key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def add(self, row: dict[str, Any]) -> None:
        """Ghi ngay từng kết quả (an toàn đa luồng): dừng giữa chừng không mất phần đã trả phí."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.data[row["key"]] = row


def run_judge(
    system: str,
    items: Sequence[dict[str, str]],
    cfg: dict[str, Any],
    *,
    cache: JudgeCache,
    gate: ApiGate,
    client: LLMClient | None = None,
    retry_errors: bool = False,
) -> list[dict[str, Any]] | None:
    """Chấm ``items`` (mỗi item: id, document, prediction). Trả kết quả theo thứ tự; None nếu thiếu --allow_api.

    retry_errors=True: chấm lại cả các mục trong cache bị lỗi parse (dòng mới ghi đè dòng cũ khi nạp cache)."""
    jcfg, client_cfg = cfg["judge"], cfg["llm"]["judge"]
    version = str(jcfg["prompt_version"])
    keyed = [(cache_key(client_cfg["model"], version, system, it["id"], it["prediction"]), it) for it in items]
    def _needs(k: str) -> bool:
        row = cache.get(k)
        row = resolve_cached(cache.get(k))
        return row is None or (retry_errors and row.get("error") is not None)

    todo = [(k, it) for k, it in keyed if _needs(k)]
    if todo:
        gate.plan(f"judge:{system}", len(todo))
        if not gate.allowed:
            print(f"[judge:{system}] BỎ QUA — cần --allow_api. Ước tính: {len(todo)} request tới "
                  f"{client_cfg['provider']}/{client_cfg['model']} ({len(keyed) - len(todo)} đã có trong cache).")
            if all(cache.get(k) is not None for k, _ in todo):   # chỉ là chấm lại mục lỗi: giữ kết quả cũ
                return [resolve_cached(cache.get(k)) for k, _ in keyed]
            return None
        client = client or LLMClient(client_cfg, label=f"judge:{system}", gate=gate)

        def _one(pair):
            key, it = pair
            prompt, truncated = build_judge_prompt(it["document"], it["prediction"], int(jcfg["max_document_chars"]))
            raw, _ = client.complete(prompt, system=SYSTEM_PROMPT)
            row = {"key": key, "system": system, "id": it["id"], "judge_model": client_cfg["model"],
                   "prompt_version": version, "document_truncated": truncated, "raw": raw}
            try:
                row.update(parse_judge_output(raw))
                row["error"] = None
            except JudgeParseError as err:
                row.update({"score": None, "unsupported_claims": [], "rationale": "", "error": str(err)})
            cache.add(row)
            return row

        print(f"[judge:{system}] Gọi API: {len(todo)} request tới {client_cfg['provider']}/{client_cfg['model']}")
        client.map(todo, _one)
    return [resolve_cached(cache.get(k)) for k, _ in keyed]
