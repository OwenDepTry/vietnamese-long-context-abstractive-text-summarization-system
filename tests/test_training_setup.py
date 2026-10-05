"""Test phần không cần GPU/model của phase 2: config, chọn LoRA target, ROUGE, guard NaN, đọc dữ liệu."""

from pathlib import Path

import pytest

from vnsum.models.lora import resolve_target_modules, summarize_module_names
from vnsum.models.rouge import VietnameseRougeTokenizer
from vnsum.models.train_config import TrainConfigError, apply_cli_overrides, deep_merge, load_train_config

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "train.yaml"


def _t5_linear_names(layers: int = 2, gated: bool = False) -> list[str]:
    """Tên Linear giống T5ForConditionalGeneration (rút gọn số lớp)."""
    ffn = ["wi_0", "wi_1", "wo"] if gated else ["wi", "wo"]
    names = []
    for i in range(layers):
        names += [f"encoder.block.{i}.layer.0.SelfAttention.{m}" for m in "qkvo"]
        names += [f"encoder.block.{i}.layer.1.DenseReluDense.{m}" for m in ffn]
        names += [f"decoder.block.{i}.layer.0.SelfAttention.{m}" for m in "qkvo"]
        names += [f"decoder.block.{i}.layer.1.EncDecAttention.{m}" for m in "qkvo"]
        names += [f"decoder.block.{i}.layer.2.DenseReluDense.{m}" for m in ffn]
    return names + ["lm_head"]


# ------------------------------- config ----------------------------------- #


def test_repo_train_config_defaults():
    cfg = load_train_config(CFG_PATH)
    assert cfg["precision"] == "fp32"  # mặc định an toàn trên T4
    assert cfg["training"]["gradient_checkpointing"] is True
    assert cfg["training"]["gradient_accumulation_steps"] > 1
    assert cfg["data"]["max_source_length"] == 1024
    assert cfg["data"]["max_target_length"] == 128
    assert cfg["training"]["save_strategy"] == "steps"
    assert cfg["resume"] == "auto"


def test_smoke_overrides_applied_only_with_limit():
    cfg = load_train_config(CFG_PATH)
    smoke = apply_cli_overrides(cfg, limit=32, max_steps=5, output_dir="tmp/run")
    assert smoke["limit"] == 32
    assert smoke["training"]["max_steps"] == 5
    assert smoke["training"]["gradient_accumulation_steps"] == 1
    assert smoke["training"]["save_steps"] == 5
    assert smoke["paths"]["output_dir"] == "tmp/run"
    assert smoke["training"]["learning_rate"] == cfg["training"]["learning_rate"]  # giữ nguyên phần không override
    full = apply_cli_overrides(cfg)
    assert full["limit"] is None and "smoke" not in full
    assert full["training"]["gradient_accumulation_steps"] == cfg["training"]["gradient_accumulation_steps"]


def test_precision_values():
    cfg = load_train_config(CFG_PATH)
    assert apply_cli_overrides(cfg, precision="bf16")["precision"] == "bf16"  # kiểm tra phần cứng lúc chạy
    with pytest.raises(TrainConfigError, match="fp8"):
        apply_cli_overrides(cfg, precision="fp8")


def test_rtx4050_config_extends_base():
    base = load_train_config(CFG_PATH)
    local = load_train_config(ROOT / "configs" / "train_rtx4050.yaml")
    assert local["precision"] == "bf16"
    assert "extends" not in local
    t, b = local["training"], base["training"]
    # Giữ batch hiệu dụng 32 như config T4 -> learning_rate vẫn hợp lệ.
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == (
        b["per_device_train_batch_size"] * b["gradient_accumulation_steps"]
    )
    assert t["learning_rate"] == b["learning_rate"]  # kế thừa
    assert local["lora"] == base["lora"]
    assert local["paths"]["output_dir"] != base["paths"]["output_dir"]


def test_extends_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(TrainConfigError, match="extends"):
        load_train_config(tmp_path / "a.yaml")


def test_hierarchical_training_rejected():
    cfg = load_train_config(CFG_PATH)
    cfg["data"]["input_strategy"] = "hierarchical"
    with pytest.raises(TrainConfigError, match="hierarchical"):
        apply_cli_overrides(cfg)


def test_training_section_cannot_set_managed_keys():
    cfg = load_train_config(CFG_PATH)
    cfg["training"]["fp16"] = True
    with pytest.raises(TrainConfigError, match="fp16"):
        apply_cli_overrides(cfg)


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


# -------------------------- LoRA target modules ----------------------------- #


def test_summarize_t5_module_names():
    summary = summarize_module_names(_t5_linear_names())
    assert summary["q"] == 2 * 3  # 2 lớp x (enc self + dec self + dec cross)
    assert summary["wi"] == 4 and "wi_0" not in summary
    assert summary["lm_head"] == 1


def test_resolve_attention_only():
    assert resolve_target_modules(_t5_linear_names(), ["q", "k", "v", "o"], include_ffn=False) == ["q", "k", "v", "o"]


@pytest.mark.parametrize("gated, ffn", [(False, ["wi", "wo"]), (True, ["wi_0", "wi_1", "wo"])])
def test_resolve_with_ffn_detects_variant(gated, ffn):
    got = resolve_target_modules(_t5_linear_names(gated=gated), ["q", "k", "v", "o"], include_ffn=True)
    assert got == ["q", "k", "v", "o"] + ffn


def test_resolve_rejects_unknown_and_lm_head():
    with pytest.raises(ValueError, match="query"):
        resolve_target_modules(_t5_linear_names(), ["query"], include_ffn=False)
    with pytest.raises(ValueError, match="lm_head"):
        resolve_target_modules(_t5_linear_names(), ["q", "lm_head"], include_ffn=False)


# ---------------------------------- ROUGE ----------------------------------- #


def test_rouge_tokenizer_keeps_vietnamese_diacritics():
    assert VietnameseRougeTokenizer().tokenize("Hà Nội, mưa TO!") == ["hà", "nội", "mưa", "to"]


def test_compute_rouge_vietnamese():
    pytest.importorskip("rouge_score")
    from vnsum.models.rouge import compute_rouge

    assert compute_rouge(["Hà Nội mưa to"], ["Hà Nội mưa to"])["rouge1"] == 100.0
    # Tokenizer mặc định của rouge_score coi "Hà" và "Ha" gần như giống nhau; ở đây phải khác.
    partial = compute_rouge(["Ha Noi mua to"], ["Hà Nội mưa to"])
    assert partial["rouge1"] == 25.0  # chỉ "to" trùng
    assert compute_rouge([], []) == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ------------------------------ guard NaN/inf ------------------------------- #


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_loss_callback_stops_with_clear_error(bad):
    pytest.importorskip("transformers")
    from transformers import TrainerControl, TrainerState

    from vnsum.models.callbacks import NonFiniteLossCallback, NonFiniteLossError

    cb = NonFiniteLossCallback(precision="fp16")
    state, control = TrainerState(), TrainerControl()
    state.global_step = 7
    cb.on_log(None, state, control, logs={"loss": 2.5})  # hữu hạn -> không làm gì
    assert not control.should_training_stop
    with pytest.raises(NonFiniteLossError) as err:
        cb.on_log(None, state, control, logs={"loss": bad})
    assert control.should_training_stop
    msg = str(err.value)
    assert "step 7" in msg and "fp32" in msg and str(bad) in msg


# ------------------------------- đọc dữ liệu -------------------------------- #


def test_read_split_selects_source_column(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from vnsum.models.data import read_split

    d = tmp_path / "vietnews"
    d.mkdir()
    pd.DataFrame(
        {"id": ["a", "b"], "document": ["doc a", "doc b"], "summary": ["s a", ""], "input_text": ["in a", "in b"]}
    ).to_parquet(d / "train.parquet")
    ext = read_split(tmp_path, "vietnews", "train", "extractive_filter")
    assert ext["source"].tolist() == ["in a"]  # summary rỗng bị bỏ
    tr = read_split(tmp_path, "vietnews", "train", "truncate")
    assert tr["source"].tolist() == ["doc a"]
    with pytest.raises(FileNotFoundError):
        read_split(tmp_path, "wikilingua_vi", "train", "truncate")
