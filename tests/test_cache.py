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


def test_model_identity_rehashes_after_overwrite_in_same_process(tmp_path):
    # Review lần 2: ghi đè trọng số tại CÙNG thư mục, tính lại trong cùng tiến trình.
    import os

    d = _model_dir(tmp_path / "m", b"weights-v1")
    before = model_identity(str(d))
    w = d / "model.safetensors"
    w.write_bytes(b"weights-v2-other")
    st = w.stat()
    os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # chắc chắn đổi mtime cả khi hệ thống tệp thô
    assert model_identity(str(d)) != before


def test_run_info_saves_diff_when_dirty(tmp_path, monkeypatch):
    import vnsum.eval.run_info as ri

    monkeypatch.setattr(ri, "git_state", lambda cwd=None: {"commit": "abc", "dirty": True, "diff": "-a\n+b"})
    out = ri.write_run_info(tmp_path / "run_info.json", argv=["x"], cfg={}, systems={}, n_samples=1)
    info = json.loads(out.read_text(encoding="utf-8"))
    assert info["git"]["diff_file"] == "run_info.diff" and "diff" not in info["git"]
    assert (tmp_path / "run_info.diff").read_text(encoding="utf-8") == "-a\n+b\n"
    monkeypatch.setattr(ri, "git_state", lambda cwd=None: {"commit": "abc", "dirty": False})
    ri.write_run_info(tmp_path / "clean" / "run_info.json", argv=["x"], cfg={}, systems={}, n_samples=1)
    assert not (tmp_path / "clean" / "run_info.diff").exists()


def test_git_state_reads_utf8_diff(tmp_path):
    # Windows: subprocess đọc bằng cp1252 mặc định -> diff tiếng Việt từng làm hỏng evaluate.py.
    import shutil
    import subprocess

    import pytest

    if not shutil.which("git"):
        pytest.skip("không có git")
    from vnsum.eval.run_info import git_state

    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
    f = tmp_path / "a.txt"
    f.write_text("Hà Nội\n", encoding="utf-8")
    run("add", "a.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "a")
    f.write_text("Đà Nẵng “nhiễu động” — ✓\n", encoding="utf-8")
    state = git_state(tmp_path)
    assert state["dirty"] is True and "Đà Nẵng" in state["diff"]


def test_run_info_never_breaks_evaluation(tmp_path, monkeypatch):
    import vnsum.eval.run_info as ri

    def boom(cwd=None):
        raise AttributeError("'NoneType' object has no attribute 'strip'")

    monkeypatch.setattr(ri, "git_state", boom)
    out = ri.write_run_info(tmp_path / "run_info.json", argv=[], cfg={}, systems={}, n_samples=0)
    assert "AttributeError" in json.loads(out.read_text(encoding="utf-8"))["git"]["error"]


def test_cache_key_includes_batch_size(tmp_path, monkeypatch):
    # Review lần 3: đổi batch_size chung (cùng hệ thống, input, checkpoint) từng trả lại output của batch cũ.
    import vnsum.eval.systems as systems

    calls = []

    def fake_generate(texts, model_path, sys_cfg, gen, token_env):
        bs = systems.effective_generation({"generation": gen}, sys_cfg)["batch_size"]
        calls.append(bs)
        return [f"batch {bs}"] * len(texts)

    monkeypatch.setattr(systems, "generate_seq2seq", fake_generate)
    sys_cfg = {"type": "seq2seq", "input_column": "input_text"}
    df = pd.DataFrame({"id": ["a"], "input_text": ["văn bản"]})
    kw = dict(results_dir=tmp_path, gate=ApiGate(allowed=False), model_path="VietAI/x")
    gen = {"num_beams": 4, "max_new_tokens": 128}
    assert run_system("s", sys_cfg, df, {"generation": {**gen, "batch_size": 8}}, **kw) == {"a": "batch 8"}
    assert run_system("s", sys_cfg, df, {"generation": {**gen, "batch_size": 4}}, **kw) == {"a": "batch 4"}
    assert run_system("s", sys_cfg, df, {"generation": {**gen, "batch_size": 8}}, **kw) == {"a": "batch 8"}
    assert calls == [8, 4]                                      # batch 8 lần hai lấy từ cache, đúng output của nó


def test_hierarchical_fingerprint_follows_data_config_content(tmp_path):
    """Review lần 4: sửa NỘI DUNG data_config mà giữ đường dẫn phải làm đổi fingerprint."""
    from vnsum.eval.systems import system_fingerprints

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data.yaml"
    data.write_text((root / "configs" / "data.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    sys_cfg = {"type": "seq2seq", "input_column": "input_text", "long_strategy": "hierarchical",
               "hierarchical": {"data_config": str(data)}}
    cfg = {"generation": {"num_beams": 4}}
    fp1, eq1 = system_fingerprints("h", sys_cfg, cfg, "m")
    assert eq1, "data_config chưa đổi: cache cũ (fingerprint không gồm data_config) vẫn dùng được"
    data.write_text(data.read_text(encoding="utf-8").replace("min_chars: 2", "min_chars: 3"), encoding="utf-8")
    fp2, eq2 = system_fingerprints("h", sys_cfg, cfg, "m")
    assert fp2 != fp1 and eq2 == []


def test_prediction_cache_accepts_equivalent_fingerprints(tmp_path):
    PredictionCache(tmp_path, "s", "old").add({"a": "x"}, {"a": "h"})
    assert PredictionCache(tmp_path, "s", "new").missing(["a"], ["h"]) == ["a"]
    c = PredictionCache(tmp_path, "s", "new", equivalent_fps=["old"])
    assert c.missing(["a"], ["h"]) == [] and c.equivalent_rows == 1
