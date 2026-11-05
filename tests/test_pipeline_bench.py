"""Benchmark đường xử lý của API (src/vnsum/inference/pipeline_bench.py, scripts/benchmark_pipeline.py)."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pandas as pd

from test_inference import FakeBackend, cfg, data_cfg  # noqa: F401
from vnsum.inference.pipeline_bench import (
    DEFAULT_VARIANTS, outputs_identical, render_report, run_ui_flow, select_mixed, summarize_flow, variant_config,
)
from vnsum.inference.predictor import Summarizer

ROOT = Path(__file__).resolve().parents[1]


def _summ(cfg, data_cfg, **ld):  # noqa: F811
    cfg = copy.deepcopy(cfg)
    cfg["max_input_tokens"] = 40
    cfg["long_document"].update(chunk_max_tokens=30, overlap_tokens=5, hierarchical_above_tokens=100,
                                reduce="concat", chunk_length="medium", **ld)
    backend = FakeBackend()
    return Summarizer(backend, cfg, data_cfg), backend, cfg


def test_select_mixed_takes_both_groups():
    df = pd.DataFrame({"id": range(20), "is_long": [i < 3 for i in range(20)]})
    out = select_mixed(df, n_short=5, n_long=10, seed=0)
    assert int(out["is_long"].sum()) == 3 and int((~out["is_long"]).sum()) == 5   # chỉ có 3 văn bản dài


def test_ui_flow_old_vs_new(cfg, data_cfg, make_doc):  # noqa: F811
    docs = [make_doc(20)[0], "Hà Nội hôm nay có mưa rào. Nhiệt độ giảm nhẹ."]
    _, _, base_cfg = _summ(cfg, data_cfg)
    res = {}
    for name, ov in DEFAULT_VARIANTS.items():
        backend = FakeBackend()
        summ = Summarizer(backend, variant_config(base_cfg, ov), data_cfg)
        res[name] = (run_ui_flow(summ, docs), backend)
    (old, b_old), (new, b_new) = res.values()
    assert len(old) == len(new) == 6 and set(old["strategy"]) == {"hierarchical", "direct"}
    # Cũ: 3 lần × mỗi chunk một lần gọi. Mới: chunk sinh theo batch một lần, 2 mức sau dùng lại.
    assert len(b_new.calls) < len(b_old.calls)
    hier_new = new[new["strategy"] == "hierarchical"]
    assert hier_new["chunk_cache_hit"].tolist() == [False, True, True]
    assert outputs_identical(old, new) == 100.0
    tables = summarize_flow(new)
    assert set(tables["per_document"].index) == {"hierarchical", "direct"}
    text = render_report({k: v[0] for k, v in res.items()}, header=["# t"], memory={"RAM": "1"})
    assert "Tổng thời gian chờ" in text and "| hiện tại (config) | 100.0 | 100.0 |" in text


def test_script_end_to_end(tmp_path, monkeypatch, cfg, data_cfg, make_doc):  # noqa: F811
    import vnsum.eval.dataset as dataset
    import vnsum.inference.predictor as predictor

    df = pd.DataFrame({"id": ["s1", "l1"], "document": ["Hà Nội mưa to. Đường ngập.", make_doc(20)[0]],
                       "is_long": [False, True]})
    monkeypatch.setattr(dataset, "load_eval_set", lambda c, limit=None: df)
    monkeypatch.setattr(predictor, "build_backend", lambda *a, **k: FakeBackend())
    spec = importlib.util.spec_from_file_location("benchmark_pipeline", ROOT / "scripts" / "benchmark_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(ROOT)
    out = tmp_path / "pb.md"
    assert mod.main(["--config", "configs/inference.yaml", "--n_short", "1", "--n_long", "1", "--warmup", "1",
                     "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "3 mức độ dài" in text and "hiện tại (config)" in text
    assert len(pd.read_csv(out.with_suffix(".csv"))) == 2 * 2 * 3


def test_ablation_has_four_variants_and_cache_never_changes_output(cfg, data_cfg, make_doc):  # noqa: F811
    from vnsum.inference.pipeline_bench import ABLATION_VARIANTS

    _, _, base_cfg = _summ(cfg, data_cfg)
    docs = [make_doc(20)[0]]
    res = {n: run_ui_flow(Summarizer(FakeBackend(), variant_config(base_cfg, ov), data_cfg), docs)
           for n, ov in ABLATION_VARIANTS.items()}
    assert len(res) == 4
    assert outputs_identical(res["batch 1, không dùng lại"], res["batch 1, dùng lại"]) == 100.0
    text = render_report(res, header=["# t"], memory={})
    assert text.count("| batch ") >= 3 and "vit5_lora_hier_concat_b1" in text
