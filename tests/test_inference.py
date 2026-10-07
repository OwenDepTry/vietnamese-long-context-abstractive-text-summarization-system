"""Phase 4: config, Summarizer (backend giả), thống kê benchmark, report, kiểm tra output.

Test tích hợp export ONNX thật (model T5 tí hon tạo ngẫu nhiên) chỉ chạy khi có optimum + torch
(tức trong .venv-onnx); ở .venv chính tự skip.
"""

from __future__ import annotations

import copy
import unicodedata
from pathlib import Path

import pytest

from conftest import WhitespaceTokenizer
from vnsum.config import load_config
from vnsum.inference.bench import (
    RUNS, bucket_edges_labels, bucket_latency, latency_stats, runs_for, select_documents, write_report,
)
from vnsum.inference.config import (
    InferenceConfigError, generation_kwargs, load_inference_config, onnx_dirs, resolve_model_path,
    validate_inference_config,
)
from vnsum.inference.export import SAMPLE_TEXT, check_vietnamese_output
from vnsum.inference.hardware import dir_size_mb, peak_rss_mb, system_info
from vnsum.inference.predictor import GenerationOutput, Summarizer

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "inference.yaml"


@pytest.fixture
def cfg():
    return load_inference_config(CFG_PATH)


@pytest.fixture
def data_cfg():
    return load_config(ROOT / "configs" / "data.yaml")


# --------------------------------------------------------------------------- config
def test_repo_config_and_length_presets(cfg):
    for name, tokens in cfg["lengths"].items():
        g = generation_kwargs(cfg, name)
        assert g["max_new_tokens"] == tokens and g["num_beams"] == 4 and g["no_repeat_ngram_size"] == 3
    assert generation_kwargs(cfg)["max_new_tokens"] == cfg["lengths"][cfg["default_length"]] == 128
    with pytest.raises(ValueError):
        generation_kwargs(cfg, "rat_dai")


def test_config_rejects_bad_values(cfg):
    for path, value in ((("backend",), "tensorrt"), (("onnx", "variant"), "int4"), (("pytorch", "dtype"), "fp16"),
                        (("default_length",), "xl"), (("long_document", "chunk_max_tokens"), 4096),
                        (("export", "quantization", "isa"), "neon")):
        bad = copy.deepcopy(cfg)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        with pytest.raises(InferenceConfigError):
            validate_inference_config(bad)


def test_paths_resolution(cfg):
    assert resolve_model_path(cfg, "D:/m/merged") == "D:/m/merged"
    assert resolve_model_path(cfg) == "outputs/vit5-vnsum-merged"          # paths.merged_dir của train.yaml
    dirs = onnx_dirs(cfg, "C:/vnsum-runs/vit5-vnsum-merged/")
    assert dirs["int8"] == Path("C:/vnsum-runs/vit5-vnsum-merged-onnx/int8")
    cfg["model"]["onnx_root"] = "E:/onnx"
    assert onnx_dirs(cfg, "x")["fp16"] == Path("E:/onnx/fp16")


# --------------------------------------------------------------------------- Summarizer
class FakeBackend:
    name, device, dtype = "fake", "cpu", "fp32"

    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()
        self.calls: list[tuple[list[str], dict]] = []

    def generate(self, texts, gen_kwargs):
        self.calls.append((list(texts), dict(gen_kwargs)))
        return GenerationOutput([f"tóm tắt {len(t.split())} từ" for t in texts], [5] * len(texts))


@pytest.fixture
def small(cfg, data_cfg):
    cfg = copy.deepcopy(cfg)
    cfg["max_input_tokens"] = 40                     # ngân sách văn bản 39 "token" (1 từ = 1 token)
    cfg["long_document"].update(chunk_max_tokens=30, overlap_tokens=5, hierarchical_above_tokens=100)
    backend = FakeBackend()
    return Summarizer(backend, cfg, data_cfg), backend, cfg


def test_direct_when_document_fits(small):
    summ, backend, cfg = small
    r = summ.summarize("Hà Nội hôm nay có mưa rào. Nhiệt độ giảm nhẹ.", length="short")
    assert r.strategy == "direct" and r.generate_calls == 1 and len(backend.calls) == 1
    assert backend.calls[0][1]["max_new_tokens"] == cfg["lengths"]["short"]
    assert r.model_input == "Hà Nội hôm nay có mưa rào. Nhiệt độ giảm nhẹ."
    assert set(r.timings_ms) >= {"preprocess_ms", "generate_ms", "total_ms"}


def test_extractive_for_moderately_long_document(small, make_doc):
    summ, backend, _ = small
    doc, _ = make_doc(6)                                          # 72 từ: > 39, ≤ 100
    r = summ.summarize(doc)
    assert r.strategy == "extractive" and len(backend.calls) == 1
    assert summ.counter.count(r.model_input) <= 39 and r.model_input_tokens <= 40
    assert backend.calls[0][1]["max_new_tokens"] == 128


def test_hierarchical_summarizes_chunks_then_combines(small, make_doc):
    summ, backend, cfg = small
    doc, _ = make_doc(20)                                         # 240 từ > 100
    r = summ.summarize(doc, length="long")
    chunk_calls, final = backend.calls[:-1], backend.calls[-1]
    assert r.strategy == "hierarchical" and len(chunk_calls) >= 2
    assert all(c[1]["max_new_tokens"] == cfg["lengths"]["short"] for c in chunk_calls)
    assert final[1]["max_new_tokens"] == cfg["lengths"]["long"]
    assert all(summ.counter.count(c[0][0]) <= 29 for c in chunk_calls)   # mỗi chunk vừa ngân sách
    assert r.generate_calls == len(backend.calls)
    # Vòng 1 tạo ~19 bản tóm tắt chunk (76 "token") > ngân sách 39 -> phải có vòng 2 trên bản ghép.
    assert 0 < len(r.chunk_summaries) < len(chunk_calls)
    assert summ.counter.count(final[0][0]) <= 39


def test_hierarchical_single_round_combines_chunk_summaries(small, make_doc):
    summ, backend, cfg = small
    cfg["long_document"]["hierarchical_above_tokens"] = 50
    doc, _ = make_doc(5)                                          # 60 từ -> 3 chunk, ghép 12 "token" vừa ngân sách
    r = summ.summarize(doc, strategy="auto")
    assert r.strategy == "hierarchical" and len(r.chunk_summaries) == len(backend.calls) - 1
    assert backend.calls[-1][0][0] == " ".join(r.chunk_summaries)


def test_hierarchical_requested_on_short_text_goes_direct(small):
    summ, backend, _ = small
    assert summ.summarize("Một câu ngắn thôi.", strategy="hierarchical").strategy == "direct"


def test_input_is_normalized_like_phase1(small):
    summ, backend, _ = small
    nfd = unicodedata.normalize("NFD", "<b>Hà Nội</b>  mưa   to.")
    r = summ.summarize(nfd)
    assert r.model_input == "Hà Nội mưa to." and unicodedata.is_normalized("NFC", r.model_input)
    with pytest.raises(ValueError):
        summ.summarize("   ")
    with pytest.raises(ValueError):
        summ.summarize("abc", length="rat_dai")


# --------------------------------------------------------------------------- benchmark logic
def test_runs_and_document_selection():
    pd = pytest.importorskip("pandas")
    assert runs_for("cpu") == ["pytorch_cpu", "onnx_fp32_cpu", "onnx_int8_cpu"]
    assert set(runs_for("all")) == set(RUNS)
    df = pd.DataFrame({"id": [str(i) for i in range(50)], "document": ["d"] * 50})
    a = select_documents(df, 10, 42)
    assert len(a) == 10 and a["id"].tolist() == select_documents(df, 10, 42)["id"].tolist()
    assert len(select_documents(df, 500, 42)) == 50


def test_latency_stats_and_buckets():
    s = latency_stats([100, 200, 300, 400], [10, 10, 10, 10], [50, 50, 50, 50])
    assert s["p50_ms"] == pytest.approx(250) and s["p95_ms"] == pytest.approx(385)
    assert s["docs_per_s"] == pytest.approx(4.0) and s["tokens_per_s"] == pytest.approx(200.0)
    assert [lab for _, _, lab in bucket_edges_labels([256, 512])] == ["≤256", "257–512", ">512"]
    b = bucket_latency([100, 200, 900], [100, 300, 1000], [256, 512])
    assert [x["n"] for x in b] == [1, 1, 1] and b[2]["p50_ms"] == 900


def _fake_result(key, lat, preds, tokens=(100, 300, 1000)):
    return {"run": key, "status": "ok", "n": len(lat), "dtype": "fp32", "latency_ms": lat, "generate_ms": [x * 0.9 for x in lat],
            "new_tokens": [20] * len(lat), "input_tokens": list(tokens), "predictions": preds, "peak_rss_mb": 1500.0,
            "disk_mb": 900.0, "threads": 8}


def test_report_flags_quality_drop_and_target(tmp_path, cfg):
    pytest.importorskip("rouge_score")
    refs = ["hà nội mưa to kéo dài", "giá xăng tăng nhẹ hôm nay", "đội tuyển thắng trận mở màn"]
    results = {
        "pytorch_cpu": _fake_result("pytorch_cpu", [1200.0, 1500.0, 2500.0], refs),
        "onnx_fp32_cpu": {"run": "onnx_fp32_cpu", "status": "skipped", "reason": "chưa có bản ONNX fp32"},
        "onnx_int8_cpu": _fake_result("onnx_int8_cpu", [300.0, 450.0, 900.0], ["hà nội mưa", "trời nắng đẹp", "không liên quan"]),
    }
    info = {"cpu": "CPU thử", "logical_cores": 8, "ram_gb": 16.0, "gpus": [], "os": "TestOS", "machine": "x86_64",
            "libraries": {"python": "3.13"}, "onnxruntime_providers": ["CPUExecutionProvider"]}
    out = tmp_path / "benchmark.md"
    warnings = write_report(results, refs, cfg, info, out, command="python scripts/benchmark.py --limit 3",
                            gen_kwargs=generation_kwargs(cfg, "medium"))
    text = out.read_text(encoding="utf-8")
    assert warnings and "onnx_int8_cpu" in warnings[0]
    assert "⚠ giảm quá ngưỡng" in text and "## Kết quả CPU" in text and "## Kết quả GPU" not in text
    assert "CPU thử" in text and "bỏ qua: chưa có bản ONNX fp32" in text
    assert "| `pytorch_cpu` | 1,500 | không đạt |" in text
    assert "`onnx_int8_cpu` với input ≤256 token" in text and "Không lượt chạy nào đạt p95 < 500 ms" in text
    assert "Chỉ có 3 tài liệu" in text
    assert "| `onnx_int8_cpu` | `pytorch_cpu` |" in text and "0/3" in text


# --------------------------------------------------------------------------- export check + phần cứng
def test_check_vietnamese_output():
    assert check_vietnamese_output("Tuyến xe buýt điện số 12 sẽ hoạt động từ đầu tháng 11, nối bến xe trung tâm.") == []
    assert "quá ngắn" in check_vietnamese_output("Hà Nội.")[0]
    assert any("dấu" in p for p in check_vietnamese_output("The new electric bus line opens in November downtown"))
    assert any("lặp" in p for p in check_vietnamese_output("xe buýt điện " * 10))
    assert any("token lỗi" in p for p in check_vietnamese_output("Tuyến xe buýt <unk> điện mới sẽ chạy"))
    assert check_vietnamese_output(SAMPLE_TEXT) == []
    garbage = ". . . c . . Đ . . đ . .nh . . v . . l . . th . . h . . b . . ch . . tr . . k . . m . ."  # output FP16 lỗi thật
    assert any("rời rạc" in p for p in check_vietnamese_output(garbage))


def test_hardware_helpers(tmp_path):
    peak = peak_rss_mb()
    assert peak is None or peak > 1
    (tmp_path / "a.onnx").write_bytes(b"0" * 1024 * 1024)
    (tmp_path / "b.txt").write_bytes(b"0" * 1024 * 1024)
    assert dir_size_mb(tmp_path, ("*.onnx",)) == pytest.approx(1.0)
    info = system_info()
    assert info["cpu"] and info["logical_cores"] and "python" in info["libraries"]


# --------------------------------------------------------------------------- tích hợp (chỉ trong .venv-onnx)
def test_tiny_t5_export_quantize_and_generate(tmp_path, cfg):
    pytest.importorskip("optimum.onnxruntime")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    from vnsum.inference.export import export_onnx, quantize_int8

    torch.manual_seed(0)
    config = transformers.T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2,
                                   num_heads=4, feed_forward_proj="relu", decoder_start_token_id=0, pad_token_id=0,
                                   eos_token_id=1)
    model = transformers.T5ForConditionalGeneration(config).eval()
    src = tmp_path / "tiny"
    model.save_pretrained(src)
    fp32, int8 = export_onnx(str(src), tmp_path / "onnx" / "fp32", cfg), tmp_path / "onnx" / "int8"
    names = sorted(p.name for p in fp32.glob("*.onnx"))
    assert names == ["decoder_model.onnx", "decoder_with_past_model.onnx", "encoder_model.onnx"]
    quantize_int8(fp32, int8, cfg)
    assert sorted(p.name for p in int8.glob("*.onnx")) == names
    assert dir_size_mb(int8, ("*.onnx",)) < dir_size_mb(fp32, ("*.onnx",))
    from vnsum.inference.export import convert_fp16

    fp16 = tmp_path / "onnx" / "fp16"
    convert_fp16(fp32, fp16)
    assert sorted(p.name for p in fp16.glob("*.onnx")) == names
    assert dir_size_mb(fp16, ("*.onnx",)) < dir_size_mb(fp32, ("*.onnx",))

    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 1]])
    kw = {"max_new_tokens": 8, "num_beams": 2, "do_sample": False}
    with torch.inference_mode():
        ref = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
    ort_fp32 = ORTModelForSeq2SeqLM.from_pretrained(fp32, use_cache=True, use_merged=False)
    out = ort_fp32.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
    assert out.tolist() == ref.tolist()                         # FP32 + KV cache: giống hệt PyTorch
    ort_int8 = ORTModelForSeq2SeqLM.from_pretrained(int8, use_cache=True, use_merged=False)
    assert ort_int8.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw).shape[0] == 1


def test_tokenizer_saved_by_transformers5_loads_in_any_version(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    import json

    from vnsum.inference.tokenizer import load_tokenizer_compat

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "xin": 3, "chào": 4}
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tk.save(str(tmp_path / "tokenizer.json"))
    # Định dạng transformers 5 (transformers 4.57 báo AttributeError với extra_special_tokens dạng list).
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "TokenizersBackend", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>",
        "extra_special_tokens": ["<extra_id_0>"]}), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 0, "eos_token_id": 1}), encoding="utf-8")
    tok = load_tokenizer_compat(tmp_path)
    assert tok.pad_token_id == 0 and tok.eos_token_id == 1
    assert tok("xin chào", add_special_tokens=False)["input_ids"] == [3, 4]
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 3}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer_compat(tmp_path)


def test_profile_instrumentation_counts_session_calls():
    import importlib.util
    from collections import defaultdict

    spec = importlib.util.spec_from_file_location("profile_onnx", ROOT / "scripts" / "profile_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Sess:
        def run(self, names, feed):
            return [sum(feed.values())]

    class Comp:
        def __init__(self):
            self.session = Sess()

    class Model:
        def __init__(self):
            self.encoder, self.decoder, self.config = Comp(), Comp(), object()

    m = Model()
    sessions = mod.find_sessions(m)
    assert set(sessions) == {"encoder", "decoder"}
    timers = defaultdict(list)
    mod.instrument(sessions, timers)
    assert m.decoder.session.run(None, {"a": 1, "b": 2}) == [3]
    m.decoder.session.run(None, {"a": 1})
    assert len(timers["decoder"]) == 2 and "encoder" not in timers
