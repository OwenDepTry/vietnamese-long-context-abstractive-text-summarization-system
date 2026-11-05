"""Hệ thống đánh giá type pipeline: chạy đúng Summarizer + configs/inference.yaml của API (backend giả)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from test_inference import FakeBackend
from vnsum.eval.config import EvalConfigError, load_eval_config, validate_eval_config
from vnsum.eval.llm_client import ApiGate

ROOT = Path(__file__).resolve().parents[1]
SYS = {"type": "pipeline", "input_column": "document", "inference_config": "configs/inference.yaml", "length": "medium"}


@pytest.fixture
def fake_summarizer(monkeypatch):
    from vnsum.config import load_config
    from vnsum.inference import predictor

    made = []

    def from_config(cfg, backend=None, **kw):
        made.append({"cfg": cfg, "backend": backend, **kw})
        return predictor.Summarizer(FakeBackend(), cfg, load_config(cfg["data_config"]))

    monkeypatch.setattr(predictor.Summarizer, "from_config", staticmethod(from_config))
    monkeypatch.chdir(ROOT)
    return made


def test_pipeline_system_runs_api_summarizer(tmp_path, fake_summarizer):
    from vnsum.eval.systems import run_system

    df = pd.DataFrame({"id": ["a", "b"], "document": ["Hà Nội hôm nay có mưa rào. Nhiệt độ giảm nhẹ.", ""]})
    preds = run_system("api", SYS, df, {"generation": {}}, results_dir=tmp_path, gate=ApiGate(allowed=False),
                       model_path="m")
    assert preds["a"].startswith("tóm tắt") and preds["b"] == ""
    made = fake_summarizer[0]
    assert made["backend"] == "pytorch" and made["model_path"] == "m"
    assert made["cfg"]["long_document"]["reduce"] == "concat"           # đúng config đang triển khai


def test_pipeline_cache_follows_inference_config(tmp_path, fake_summarizer):
    from vnsum.eval.systems import run_system

    df = pd.DataFrame({"id": ["a"], "document": ["Hà Nội hôm nay có mưa rào."]})
    kw = dict(results_dir=tmp_path, gate=ApiGate(allowed=False), model_path="m")
    run_system("api", SYS, df, {"generation": {}}, **kw)
    run_system("api", SYS, df, {"generation": {}}, **kw)
    assert len(fake_summarizer) == 1                                    # lần hai: cache
    run_system("api", {**SYS, "long_document": {"chunk_batch_size": 1}}, df, {"generation": {}}, **kw)
    assert len(fake_summarizer) == 2                                    # đổi config suy luận: sinh lại
    assert fake_summarizer[1]["cfg"]["long_document"]["chunk_batch_size"] == 1


def test_eval_config_pipeline_system():
    cfg = load_eval_config(ROOT / "configs" / "eval.yaml")
    assert cfg["systems"]["vit5_lora_api"]["type"] == "pipeline"
    del cfg["systems"]["vit5_lora_api"]["inference_config"]
    with pytest.raises(EvalConfigError):
        validate_eval_config(cfg)
