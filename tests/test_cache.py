"""Cache dự đoán phải gắn với nội dung input và checkpoint thật (lỗi được tái hiện trong review)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vnsum.eval.llm_client import ApiGate
from vnsum.eval.systems import PredictionCache, model_identity, run_system, system_fingerprint

LEAD = {"type": "lead", "input_column": "document", "num_sentences": 1}


def _df(doc: str) -> pd.DataFrame:
    return pd.DataFrame({"id": ["x1"], "document": [doc]})


def test_lead_cache_invalidated_when_input_changes(tmp_path):
    # Tái hiện review: cùng id, nội dung đổi -> trước đây vẫn trả câu về Hà Nội.
    kw = dict(results_dir=tmp_path, gate=ApiGate(allowed=False))
    assert run_system("lead3", LEAD, _df("Hà Nội hôm nay có mưa lớn."), {}, **kw) == {"x1": "Hà Nội hôm nay có mưa lớn."}
    assert run_system("lead3", LEAD, _df("Đà Nẵng hôm nay trời nắng."), {}, **kw) == {"x1": "Đà Nẵng hôm nay trời nắng."}
    # Dữ liệu cũ quay lại -> dùng lại được dòng tương ứng (không phải dòng mới nhất).
    assert run_system("lead3", LEAD, _df("Hà Nội hôm nay có mưa lớn."), {}, **kw) == {"x1": "Hà Nội hôm nay có mưa lớn."}


def test_legacy_rows_only_reused_for_paid_api(tmp_path):
    (tmp_path / "s.jsonl").write_text(json.dumps({"id": "a", "prediction": "cũ", "fp": "f"}) + "\n", encoding="utf-8")
    strict = PredictionCache(tmp_path, "s", "f")
    assert strict.missing(["a"], ["h1"]) == ["a"]            # GPU/lead: sinh lại cho chắc
    paid = PredictionCache(tmp_path, "s", "f", accept_legacy=True)
    assert paid.missing(["a"], ["h1"]) == [] and paid.legacy_used == 1   # API trả phí: dùng lại + cảnh báo
    paid.add({"a": "mới"}, {"a": "h2"})
    again = PredictionCache(tmp_path, "s", "f", accept_legacy=True)
    assert again.missing(["a"], ["h2"]) == [] and again.data["a"] == "mới" and again.legacy_used == 0


def _model_dir(path: Path, weights: bytes) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "t5"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(weights)
    return path


def test_model_identity_follows_weights_not_path(tmp_path):
    a = _model_dir(tmp_path / "a", b"weights-v1")
    b = _model_dir(tmp_path / "moved", b"weights-v1")
    c = _model_dir(tmp_path / "c", b"weights-v2")
    assert model_identity(str(a)) == model_identity(str(b))   # chuyển thư mục: vẫn dùng được cache
    assert model_identity(str(a)) != model_identity(str(c))   # ghi đè trọng số: cache cũ bị bỏ
    assert model_identity("VietAI/vit5-base") == "VietAI/vit5-base"   # model trên Hub: giữ tên
    sys_cfg = {"type": "seq2seq", "input_column": "input_text"}
    cfg = {"generation": {"num_beams": 4}}
    assert system_fingerprint("s", sys_cfg, cfg, str(a)) != system_fingerprint("s", sys_cfg, cfg, str(c))
