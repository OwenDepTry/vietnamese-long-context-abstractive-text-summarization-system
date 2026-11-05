"""Model card, scripts/push_to_hub.py (chạy thử) và demo/space/start.py — offline, không gọi Hub."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vnsum.model_card import build_model_card, load_metrics

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_model_card_uses_reported_numbers():
    card = build_model_card(load_metrics(ROOT / "results" / "metrics.csv"), "user/vit5-vnsum", has_onnx=True)
    assert card.startswith("---\nlanguage:\n- vi")
    assert "base_model: VietAI/vit5-base" in card and "pipeline_tag: summarization" in card
    # Cùng số với results/report.md: ROUGE-L 26.98 (model này), 23.65 (ViT5 công khai), faithfulness 4.01.
    assert "| **Model này** (ViT5-base + LoRA) | 33.46 | 18.42 | 26.98 | 58.68 | 4.01 |" in card
    assert "| VietAI/vit5-base-vietnews-summarization | 29.88 | 16.24 | 23.65 |" in card
    assert "| VietNews | 1,000 |" in card and "1,910 mẫu test" in card
    assert 'snapshot_download("user/vit5-vnsum"' in card and "onnx/int8/" in card


def test_push_to_hub_dry_run_does_not_upload(tmp_path, capsys):
    model = tmp_path / "merged"
    model.mkdir()
    for name in ("config.json", "model.safetensors", "spiece.model", "notes.txt"):
        (model / name).write_bytes(b"x" * 10)
    mod = _load("scripts/push_to_hub.py", "push_to_hub")
    card = tmp_path / "card.md"
    assert mod.main(["--repo_id", "u/m", "--model_dir", str(model), "--metrics", str(ROOT / "results" / "metrics.csv"),
                     "--card_out", str(card)]) == 0
    out = capsys.readouterr().out
    assert "CHƯA upload" in out and "private" in out and "notes.txt" not in out
    assert card.read_text(encoding="utf-8").startswith("---")
    (model / "model.safetensors").unlink()
    with pytest.raises(SystemExit):                       # thư mục chưa merge: dừng sớm
        mod.main(["--repo_id", "u/m", "--model_dir", str(model), "--card_out", str(card)])


def test_demo_prepares_model_layout(tmp_path):
    start = _load("demo/space/start.py", "space_start")
    calls = []

    def fake_download(repo_id, local_dir, allow_patterns=None, ignore_patterns=None):
        calls.append((repo_id, allow_patterns, ignore_patterns))
        d = Path(local_dir)
        if allow_patterns:                                 # lần tải ONNX
            (d / "onnx" / "int8").mkdir(parents=True)
            (d / "onnx" / "int8" / "encoder_model.onnx").write_bytes(b"x")
        else:
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"x")

    model_dir, has_onnx = start.prepare_model("u/m", tmp_path, download=fake_download)
    assert has_onnx and (tmp_path / "vit5-vnsum-merged-onnx" / "int8" / "encoder_model.onnx").exists()
    assert not (tmp_path / "_onnx_download").exists()
    # Bố cục khớp với cách API tìm bản ONNX.
    from vnsum.inference.config import onnx_dirs

    assert onnx_dirs({"model": {}}, str(model_dir))["int8"] == tmp_path / "vit5-vnsum-merged-onnx" / "int8"
    env = start.api_env(model_dir, has_onnx, base={})
    assert env["VNSUM_BACKEND"] == "onnx" and env["HF_HUB_OFFLINE"] == "1"
    assert start.api_env(model_dir, False, base={})["VNSUM_BACKEND"] == "pytorch"
