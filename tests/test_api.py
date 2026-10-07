"""Phase 5: API với Summarizer thật nhưng backend giả (không cần model, torch hay onnxruntime)."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import pytest

from conftest import WhitespaceTokenizer
from vnsum.api.settings import ApiSettings, api_config
from vnsum.api.streaming import StreamingBackend, sse_event
from vnsum.config import load_config
from vnsum.inference.config import load_inference_config
from vnsum.inference.predictor import GenerationOutput, Summarizer

ROOT = Path(__file__).resolve().parents[1]


class FakeStreamer:
    """Cùng giao thức với TextIteratorStreamer: put / end / lặp để đọc."""

    def __init__(self, tokenizer=None):
        self.q: queue.Queue = queue.Queue()

    def put(self, text):
        self.q.put(text)

    def end(self):
        self.q.put(None)

    def __iter__(self):
        while (item := self.q.get(timeout=5)) is not None:
            yield item


class FakeBackend:
    def __init__(self, name="fake", fail=False, delay=0.0):
        self.name, self.device, self.dtype = name, "cpu", "fp32"
        self.tokenizer = WhitespaceTokenizer()
        self.calls: list[dict] = []
        self.fail, self.delay = fail, delay

    def generate(self, texts, gen_kwargs):
        self.calls.append(dict(gen_kwargs))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("GPU hết bộ nhớ")
        out = f"Tóm tắt {gen_kwargs['max_new_tokens']} token của {len(texts[0].split())} từ."
        if "streamer" in gen_kwargs:
            for w in out.split(" "):
                gen_kwargs["streamer"].put(w + " ")
            gen_kwargs["streamer"].end()
        return GenerationOutput([out], [7])


@pytest.fixture
def cfg():
    c = load_inference_config(ROOT / "configs" / "inference.yaml")
    c["api"] = {"max_input_chars": 2000, "max_concurrency": 1, "stream_backend": "same"}
    return c


@pytest.fixture
def data_cfg():
    return load_config(ROOT / "configs" / "data.yaml")


def _client(cfg, summ, **kw):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from vnsum.api.app import create_app

    app = create_app(ApiSettings(), cfg, summarizer_factory=lambda: summ, streamer_factory=FakeStreamer, **kw)
    return TestClient(app)


def _sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# --------------------------------------------------------------------------- phần không cần FastAPI
def test_sse_event_format():
    assert sse_event("token", {"text": "Hà Nội"}) == 'event: token\ndata: {"text": "Hà Nội"}\n\n'


def test_streaming_backend_forces_greedy_and_relays_tokens():
    events = []
    base = FakeBackend()
    sb = StreamingBackend(base, events.append, FakeStreamer)
    out = sb.generate(["một hai ba"], {"num_beams": 4, "max_new_tokens": 48, "early_stopping": True})
    assert base.calls[0]["num_beams"] == 1 and "early_stopping" not in base.calls[0]
    assert "".join(e["text"] for e in events).strip() == out.texts[0]
    assert {e["call"] for e in events} == {0}
    with pytest.raises(ValueError):
        sb.generate(["a", "b"], {"max_new_tokens": 1})


def test_streaming_backend_propagates_errors_without_hanging():
    sb = StreamingBackend(FakeBackend(fail=True), lambda e: None, FakeStreamer)
    t = time.perf_counter()
    with pytest.raises(RuntimeError, match="hết bộ nhớ"):
        sb.generate(["x"], {"max_new_tokens": 1})
    assert time.perf_counter() - t < 3


def test_api_config_validation(cfg):
    assert api_config(cfg, ApiSettings())["max_input_chars"] == 2000
    assert api_config(cfg, ApiSettings(stream_backend="pytorch"))["stream_backend"] == "pytorch"
    with pytest.raises(ValueError):
        api_config(cfg, ApiSettings(stream_backend="tensorrt"))


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("VNSUM_MODEL_PATH", "/models/m")
    monkeypatch.setenv("VNSUM_BACKEND", "onnx")
    monkeypatch.delenv("VNSUM_DEVICE", raising=False)
    s = ApiSettings.from_env()
    assert s.model_path == "/models/m" and s.backend == "onnx" and s.device is None


def test_request_id_filter_adds_id():
    pytest.importorskip("fastapi")
    from vnsum.api import app as from_app

    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    token = from_app.request_id_var.set("abc123")
    try:
        assert from_app.RequestIdFilter().filter(rec) and rec.request_id == "abc123"
    finally:
        from_app.request_id_var.reset(token)


# --------------------------------------------------------------------------- endpoint
def test_health_and_model_loaded_once(cfg, data_cfg):
    backend = FakeBackend()
    loads = []

    def factory():
        loads.append(1)
        return Summarizer(backend, cfg, data_cfg)

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from vnsum.api.app import create_app

    with TestClient(create_app(ApiSettings(), cfg, summarizer_factory=factory, streamer_factory=FakeStreamer)) as c:
        r = c.get("/health")
        c.get("/health")
    assert r.status_code == 200 and loads == [1]
    body = r.json()
    assert body["status"] == "ok" and body["backend"]["name"] == "fake" and body["max_input_chars"] == 2000
    assert body["lengths"] == {"short": 48, "medium": 128, "long": 256}


def test_summarize_returns_latency_and_respects_length(cfg, data_cfg):
    backend = FakeBackend()
    with _client(cfg, Summarizer(backend, cfg, data_cfg)) as c:
        r = c.post("/summarize", json={"text": "Hà Nội hôm nay mưa to. Nhiều tuyến phố ngập nước.", "length": "short"},
                   headers={"X-Request-ID": "req-42"})
        r2 = c.post("/summarize", json={"text": "Một câu ngắn."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "req-42" and r.headers["X-Request-ID"] == "req-42"
    assert body["length"] == "short" and body["strategy"] == "direct" and body["latency_ms"] >= 0
    assert body["summary"].startswith("Tóm tắt 48 token") and body["backend"]["name"] == "fake"
    assert backend.calls[0]["max_new_tokens"] == 48 and backend.calls[0]["num_beams"] == 4
    assert r2.json()["length"] == "medium" and len(r2.headers["X-Request-ID"]) == 12   # mặc định + id tự sinh


@pytest.mark.parametrize("payload", [
    {"text": "   "},                                   # rỗng sau khi bỏ khoảng trắng
    {"text": "a " * 1500},                             # > max_input_chars (2000)
    {"text": "văn bản", "length": "rat_dai"},          # length ngoài short/medium/long
    {"text": "văn bản", "model": "gpt"},               # trường lạ
    {"length": "short"},                               # thiếu text
])
def test_summarize_validation_errors(cfg, data_cfg, payload):
    backend = FakeBackend()
    with _client(cfg, Summarizer(backend, cfg, data_cfg)) as c:
        r = c.post("/summarize", json=payload)
    assert r.status_code == 422 and backend.calls == []


def test_stream_emits_tokens_then_done(cfg, data_cfg):
    backend = FakeBackend()
    with _client(cfg, Summarizer(backend, cfg, data_cfg)) as c:
        r = c.post("/summarize/stream", json={"text": "Giá xăng giảm nhẹ từ chiều nay.", "length": "long"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _sse(r.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("token") >= 3
    done = events[-1][1]
    assert "".join(d["text"] for k, d in events if k == "token").strip() == done["summary"]
    assert done["decoding"] == "greedy" and done["latency_ms"] >= 0 and done["request_id"] == events[0][1]["request_id"]
    assert backend.calls[0]["num_beams"] == 1 and backend.calls[0]["max_new_tokens"] == 256


def test_stream_reports_error_event(cfg, data_cfg):
    with _client(cfg, Summarizer(FakeBackend(fail=True), cfg, data_cfg)) as c:
        r = c.post("/summarize/stream", json={"text": "Văn bản thử."})
    kinds = [k for k, _ in _sse(r.text)]
    assert kinds[0] == "start" and kinds[-1] == "error" and "done" not in kinds


def test_stream_uses_pytorch_backend_when_configured(cfg, data_cfg):
    cfg["api"]["stream_backend"] = "pytorch"
    onnx, torch_like = FakeBackend(name="onnx"), FakeBackend(name="pytorch")
    with _client(cfg, Summarizer(onnx, cfg, data_cfg),
                 stream_summarizer_factory=lambda: Summarizer(torch_like, cfg, data_cfg)) as c:
        h = c.get("/health").json()
        c.post("/summarize/stream", json={"text": "Văn bản thử."})
        c.post("/summarize", json={"text": "Văn bản thử."})
    assert h["backend"]["name"] == "onnx" and h["stream_backend"]["name"] == "pytorch"
    assert len(torch_like.calls) == 1 and "streamer" in torch_like.calls[0]
    assert len(onnx.calls) == 1 and "streamer" not in onnx.calls[0]


def test_inference_runs_off_the_event_loop(cfg, data_cfg):
    """Một request chậm không chặn /health (inference chạy trong threadpool)."""
    backend = FakeBackend(delay=1.0)
    with _client(cfg, Summarizer(backend, cfg, data_cfg)) as c:
        done = {}
        t = threading.Thread(target=lambda: done.setdefault("r", c.post("/summarize", json={"text": "Chậm."})))
        t.start()
        time.sleep(0.2)
        start = time.perf_counter()
        assert c.get("/health").status_code == 200
        health_ms = (time.perf_counter() - start) * 1000
        t.join()
    assert done["r"].status_code == 200 and health_ms < 500


# --------------------------------------------------------------------------- UI (chỉ phần gọi HTTP)
def _load_ui():
    import importlib.util

    spec = importlib.util.spec_from_file_location("vnsum_ui", ROOT / "ui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ui_client_calls_api_and_formats_errors():
    httpx = pytest.importorskip("httpx")
    ui = _load_ui()
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        if body["length"] == "long":
            return httpx.Response(422, json={"detail": [{"msg": "String should have at most 50000 characters"}]})
        return httpx.Response(200, json={"summary": "ok", "latency_ms": 12.0, "new_tokens": 3, "strategy": "direct"})

    client = httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))
    assert ui.get_health(client) == {"status": "ok"}
    assert ui.summarize(client, "Văn bản", "short")["summary"] == "ok"
    err = ui.summarize(client, "Văn bản", "long")["error"]
    assert "422" in err and "at most 50000" in err
    assert json.loads(seen[1][2]) == {"text": "Văn bản", "length": "short"}

    def down(request):
        raise httpx.ConnectError("refused")

    dead = httpx.Client(base_url="http://api", transport=httpx.MockTransport(down))
    assert ui.get_health(dead) is None and "Không gọi được API" in ui.summarize(dead, "x", "short")["error"]
