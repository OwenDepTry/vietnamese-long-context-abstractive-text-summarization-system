"""Bước mở rộng B: dữ liệu distillation và trộn vào tập train (không cần GPU / API)."""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from vnsum.data.distill import build_prompt, clean_summary, estimate_cost, filter_summaries, sample_train  # noqa: E402
from vnsum.models.data import mix_distill  # noqa: E402
from vnsum.models.train_config import TrainConfigError, apply_cli_overrides, load_train_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("raw, want", [
    ("Tóm tắt: Hà Nội mưa to.", "Hà Nội mưa to."),
    ("**Bản tóm tắt:** Hà Nội mưa to.", "Hà Nội mưa to."),
    ("\u201cHà Nội mưa to.\u201d", "Hà Nội mưa to."),
    ("- Hà Nội mưa to.\n- Nhiều phố ngập.", "Hà Nội mưa to. Nhiều phố ngập."),
    ("  Hà Nội   mưa\nto. ", "Hà Nội mưa to."),
])
def test_clean_summary(raw, want):
    assert clean_summary(raw) == want


def test_filter_summaries_reasons():
    df = pd.DataFrame({"id": list("abcde"), "distill_summary": [
        "", "Xin lỗi, tôi không thể tóm tắt.", "Quá ngắn.", "từ " * 40, "Văn bản nói về " + "nội dung chính " * 6]})
    kept, dropped = filter_summaries(df, lambda texts: [len(t.split()) for t in texts], max_target_tokens=30, min_syllables=5)
    assert kept["id"].tolist() == ["e"]
    assert dropped == {"rỗng": 1, "từ chối": 1, "< 5 âm tiết": 1, "> 30 token": 1}
    assert kept["distill_tokens"].iloc[0] == len(kept["distill_summary"].iloc[0].split())


def test_build_prompt_and_estimate():
    prompt = build_prompt("Tóm tắt:\n{document}", "x" * 50, max_chars=10)
    assert prompt == "Tóm tắt:\n" + "x" * 10
    est = estimate_cost([prompt] * 4, {"chars_per_token": 2.0, "input_per_mtok": 1.0, "output_per_mtok": 2.0}, 100)
    assert est["requests"] == 4 and est["input_tokens"] == pytest.approx(len(prompt) * 4 / 2)
    assert est["usd"] == pytest.approx(est["input_tokens"] / 1e6 + 400 * 2 / 1e6)


def _fake_parquet(monkeypatch, frames):
    """Thay pd.read_parquet bằng bảng trong bộ nhớ (test chạy được khi không có pyarrow)."""
    def read(path, columns=None):
        df = frames[Path(path).as_posix()]
        return df[columns].copy() if columns else df.copy()
    monkeypatch.setattr(pd, "read_parquet", read)


def test_sample_train_is_deterministic(tmp_path, monkeypatch):
    d = tmp_path / "vietnews"
    d.mkdir()
    (d / "train.parquet").write_bytes(b"")
    _fake_parquet(monkeypatch, {(d / "train.parquet").as_posix(): pd.DataFrame({
        "id": range(50), "input_text": [f"văn bản {i}" for i in range(50)], "summary": "s", "document_tokens": 10})})
    a = sample_train(tmp_path, {"vietnews": 10}, 42)
    b = sample_train(tmp_path, {"vietnews": 10}, 42)
    assert a["id"].tolist() == b["id"].tolist() and len(a) == 10 and set(a["dataset"]) == {"vietnews"}


def test_mix_distill_excludes_conflicting_originals(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"")
    _fake_parquet(monkeypatch, {path.as_posix(): pd.DataFrame(
        {"id": ["1", "2"], "dataset": "vietnews", "input_text": ["a", "b"], "summary": ["LLM 1", "LLM 2"]})})
    original = pd.DataFrame({"id": [str(i) for i in range(1, 11)], "dataset": "vietnews",
                             "source": [f"src {i}" for i in range(1, 11)], "summary": "gốc"})
    out = mix_distill(original, {"path": str(path), "original_samples": 5}, seed=42)
    assert (out["origin"] == "distill").sum() == 2 and (out["origin"] == "original").sum() == 5
    orig_ids = set(out.loc[out["origin"] == "original", "id"])
    assert not orig_ids & {"1", "2"}                                     # không có hai nhãn cho cùng input
    assert set(out.loc[out["origin"] == "distill", "summary"]) == {"LLM 1", "LLM 2"}


def test_train_distill_config():
    cfg = apply_cli_overrides(load_train_config(ROOT / "configs" / "train_distill.yaml"), init_adapter="adapter_dir")
    assert cfg["precision"] == "bf16" and cfg["model"]["init_adapter"] == "adapter_dir"
    assert cfg["data"]["distill"]["original_samples"] == 5000
    assert cfg["training"]["learning_rate"] < 5e-4
    cfg["data"]["distill"] = {"original_samples": 10}
    with pytest.raises(TrainConfigError):
        apply_cli_overrides(cfg)
