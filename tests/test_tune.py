"""Bước 2 — tune độ dài (src/vnsum/inference/tune.py, scripts/tune_lengths.py) — chạy offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from vnsum.inference.tune import candidate_grid, candidate_name, gen_kwargs_for, pick_best, summarize_scores

ROOT = Path(__file__).resolve().parents[1]
BASE = {"num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True, "max_new_tokens": 128}


def test_grid_puts_baseline_first_and_covers_all():
    g = candidate_grid([1.0, 2.0], [0, 16])
    assert g[0] == {"length_penalty": 1.0, "min_new_tokens": 0} and len(g) == 4
    assert candidate_name(g[0]) == "lp1_min0"


def test_baseline_kwargs_identical_to_current_config():
    assert gen_kwargs_for(BASE, {"length_penalty": 1.0, "min_new_tokens": 0}) == BASE


def test_candidate_kwargs():
    k = gen_kwargs_for(BASE, {"length_penalty": 1.5, "min_new_tokens": 24})
    assert k["length_penalty"] == 1.5 and k["min_new_tokens"] == 24 and k["max_new_tokens"] == 128
    greedy = gen_kwargs_for({**BASE, "num_beams": 1}, {"length_penalty": 2.0, "min_new_tokens": 0})
    assert "length_penalty" not in greedy                     # length_penalty vô nghĩa khi greedy
    with pytest.raises(ValueError):
        gen_kwargs_for(BASE, {"length_penalty": 1.0, "min_new_tokens": 128})


def _rows():
    data = []
    for cand, rl, rep in [("lp1_min0", 25.0, 0.1), ("lp1.5_min16", 27.0, 0.5), ("lp2_min32", 29.0, 3.0)]:
        for ds in ("vietnews", "wikilingua_vi"):
            data.append({"candidate": cand, "id": f"{ds}-1", "dataset": ds, "rouge1": rl + 5, "rouge2": rl - 10,
                         "rougeL": rl, "length_syllables": 30.0, "repetition_3gram": rep})
    return pd.DataFrame(data)


def test_pick_best_rejects_repetitive_candidate():
    table = summarize_scores(_rows())
    assert list(table.index) == ["lp1_min0", "lp1.5_min16", "lp2_min32"]
    assert "rougeL_vietnews" in table.columns and table.loc["lp1_min0", "n"] == 2
    assert pick_best(table, "lp1_min0", max_repetition_increase=1.0) == "lp1.5_min16"   # lp2 lặp quá nhiều
    assert pick_best(table, "lp1_min0", max_repetition_increase=5.0) == "lp2_min32"


def test_script_end_to_end_with_fake_backend(tmp_path, monkeypatch):
    pytest.importorskip("rouge_score")
    import vnsum.eval.dataset as dataset
    import vnsum.inference.predictor as predictor
    from vnsum.inference.predictor import GenerationOutput

    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"], "dataset": ["vietnews", "vietnews", "wikilingua_vi", "wikilingua_vi"],
        "input_text": ["x"] * 4, "summary": ["một hai ba bốn năm sáu bảy tám"] * 4,
    })
    calls = []

    class Fake:
        name, dtype, device = "fake", "fp32", "cpu"

        def generate(self, texts, gk):
            calls.append(gk)
            n = 8 if gk.get("min_new_tokens") else 3              # ứng viên có min_new_tokens sinh dài hơn
            return GenerationOutput([" ".join("một hai ba bốn năm sáu bảy tám".split()[:n])] * len(texts), [n] * len(texts))

    monkeypatch.setattr(dataset, "load_eval_set", lambda cfg, limit=None: df)
    monkeypatch.setattr(predictor, "build_backend", lambda *a, **k: Fake())
    spec = importlib.util.spec_from_file_location("tune_lengths", ROOT / "scripts" / "tune_lengths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "tune.md"
    monkeypatch.chdir(ROOT)
    assert mod.main(["--config", "configs/inference.yaml", "--n", "4", "--batch_size", "2",
                     "--length_penalties", "1.0", "--min_new_tokens", "0,16", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "`lp1_min16` **(chọn)**" in text and "min_new_tokens: 16" in text
    assert (tmp_path / "tune.csv").exists() and (tmp_path / "tune" / "lp1_min0.jsonl").exists()
    n_calls = len(calls)
    mod.main(["--config", "configs/inference.yaml", "--n", "4", "--length_penalties", "1.0",
              "--min_new_tokens", "0,16", "--out", str(out)])
    assert len(calls) == n_calls                              # chạy lại: lấy từ cache, không sinh lại


def test_eval_length_kwargs_and_system_override():
    from vnsum.eval.config import load_eval_config
    from vnsum.eval.systems import _length_kwargs

    gen = {"num_beams": 4, "max_new_tokens": 128, "min_new_tokens": 24, "length_penalty": 2.0}
    assert _length_kwargs(gen, 128) == {"min_new_tokens": 24, "length_penalty": 2.0}
    assert _length_kwargs(gen, 16) == {"length_penalty": 2.0}                  # chunk ngắn: bỏ min
    assert _length_kwargs({"num_beams": 4}, 128) == {}                           # cấu hình cũ: không đổi gì
    cfg = load_eval_config(ROOT / "configs" / "eval.yaml")
    assert cfg["systems"]["vit5_lora_tuned"]["enabled"] is False
    assert "min_new_tokens" not in cfg["generation"]                           # hệ thống khác giữ nguyên
