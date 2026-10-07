"""FastAPI cho Summarizer (phase 4).

    uvicorn --factory vnsum.api.app:create_app --host 0.0.0.0 --port 8000

* Model được nạp MỘT lần trong lifespan (khi server khởi động), không nạp theo request.
* Inference là CPU/GPU-bound nên chạy trong threadpool (``run_in_threadpool``); handler async không
  block event loop. Số request inference chạy đồng thời giới hạn bởi ``api.max_concurrency``.
* Mỗi request có request id (header ``X-Request-ID`` nếu client gửi, không thì tự sinh), được ghi vào
  log và trả lại trong header + body.
"""


import asyncio
import contextvars
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, create_model
from starlette.concurrency import run_in_threadpool

from vnsum.api.settings import ApiSettings, api_config
from vnsum.api.streaming import StreamingBackend, sse_event
from vnsum.inference.config import load_inference_config

logger = logging.getLogger("vnsum.api")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


# --------------------------------------------------------------------------- logging
class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(getattr(h, "_vnsum", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler._vnsum = True  # type: ignore[attr-defined]
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _with_request_id(rid: str, fn: Callable, *args, **kwargs):
    """Chạy fn trong threadpool mà log vẫn mang request id."""
    def run():
        request_id_var.set(rid)
        return fn(*args, **kwargs)

    return run


# --------------------------------------------------------------------------- schema
class BackendInfo(BaseModel):
    name: str
    dtype: str
    device: str


class SummarizeResponse(BaseModel):
    request_id: str
    summary: str
    length: str
    strategy: str = Field(description="direct | extractive | hierarchical")
    latency_ms: float = Field(description="thời gian xử lý (chuẩn hóa + chọn câu + sinh), không gồm thời gian chờ hàng đợi")
    input_tokens: int
    model_input_tokens: int
    new_tokens: int
    generate_calls: int
    backend: BackendInfo


def _backend_info(backend: Any) -> BackendInfo:
    return BackendInfo(name=str(backend.name), dtype=str(backend.dtype), device=str(backend.device))


def build_summarizer(cfg: dict[str, Any], settings: ApiSettings, backend: str | None = None):
    from vnsum.inference.predictor import Summarizer

    return Summarizer.from_config(cfg, backend or settings.backend, model_path=settings.model_path,
                                  variant=settings.onnx_variant, device=settings.device)


# --------------------------------------------------------------------------- app
def create_app(settings: ApiSettings | None = None, cfg: dict[str, Any] | None = None, *,
               summarizer_factory: Callable[[], Any] | None = None,
               stream_summarizer_factory: Callable[[], Any] | None = None,
               streamer_factory: Callable[[Any], Any] | None = None) -> FastAPI:
    """Tạo app. Các factory cho phép test thay Summarizer thật bằng bản giả (không cần model)."""
    settings = settings or ApiSettings.from_env()
    setup_logging(settings.log_level)
    cfg = cfg or load_inference_config(settings.config_path)
    acfg = api_config(cfg, settings)
    lengths = tuple(cfg["lengths"])
    length_type = Literal[lengths]  # type: ignore[valid-type]
    text_type = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=int(acfg["max_input_chars"]))]
    SummarizeRequest = create_model(  # noqa: N806 - schema sinh từ config (độ dài tối đa, các mức độ dài)
        "SummarizeRequest",
        __config__=ConfigDict(extra="forbid"),
        text=(text_type, Field(description=f"văn bản tiếng Việt, tối đa {acfg['max_input_chars']:,} ký tự")),
        length=(length_type, Field(default=cfg["default_length"], description=f"một trong {list(lengths)}")),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t = time.perf_counter()
        summ = await run_in_threadpool(summarizer_factory or (lambda: build_summarizer(cfg, settings)))
        stream_summ = summ
        if acfg["stream_backend"] == "pytorch" and summ.backend.name != "pytorch":
            factory = stream_summarizer_factory or (lambda: build_summarizer(cfg, settings, backend="pytorch"))
            stream_summ = await run_in_threadpool(factory)
        app.state.summarizer = summ
        app.state.stream_summarizer = stream_summ
        app.state.semaphore = asyncio.Semaphore(int(acfg["max_concurrency"]))
        b = summ.backend
        logger.info("Đã nạp model trong %.1f s: %s %s %s (stream: %s)", time.perf_counter() - t,
                    b.name, b.dtype, b.device, stream_summ.backend.name)
        yield
        app.state.summarizer = app.state.stream_summarizer = None

    app = FastAPI(title="vnsum — tóm tắt văn bản tiếng Việt", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            # Với /summarize/stream, thời gian này là tới khi bắt đầu stream (log "stream ..." có tổng thời gian).
            logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
                        (time.perf_counter() - t) * 1000)
            return response
        finally:
            request_id_var.reset(token)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        summ = request.app.state.summarizer
        return {
            "status": "ok",
            "backend": _backend_info(summ.backend).model_dump(),
            "stream_backend": _backend_info(request.app.state.stream_summarizer.backend).model_dump(),
            "lengths": {k: int(v) for k, v in cfg["lengths"].items()},
            "default_length": cfg["default_length"],
            "max_input_chars": int(acfg["max_input_chars"]),
        }

    @app.post("/summarize", response_model=SummarizeResponse)
    async def summarize(req: SummarizeRequest, request: Request) -> SummarizeResponse:  # type: ignore[valid-type]
        rid = request_id_var.get()
        summ = request.app.state.summarizer
        async with request.app.state.semaphore:
            t = time.perf_counter()
            try:
                r = await run_in_threadpool(_with_request_id(rid, summ.summarize, req.text, length=req.length))
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err)) from None
            latency = (time.perf_counter() - t) * 1000
        logger.info("summarize length=%s strategy=%s input_tokens=%d new_tokens=%d latency_ms=%.0f",
                    r.length, r.strategy, r.input_tokens, r.new_tokens, latency)
        return SummarizeResponse(
            request_id=rid, summary=r.summary, length=r.length, strategy=r.strategy, latency_ms=round(latency, 1),
            input_tokens=r.input_tokens, model_input_tokens=r.model_input_tokens, new_tokens=r.new_tokens,
            generate_calls=r.generate_calls, backend=_backend_info(summ.backend),
        )

    @app.post("/summarize/stream")
    async def summarize_stream(req: SummarizeRequest, request: Request) -> StreamingResponse:  # type: ignore[valid-type]
        """SSE: ``start`` -> nhiều ``token`` (``call`` = lần sinh thứ mấy) -> ``done`` (hoặc ``error``).

        Văn bản cần hierarchical có nhiều lần sinh: các ``call`` đầu là tóm tắt từng chunk, ``call``
        cuối là bản tóm tắt cuối cùng (cũng nằm trong ``done.summary``).
        """
        rid = request_id_var.get()
        from vnsum.inference.predictor import Summarizer

        base = request.app.state.stream_summarizer
        semaphore: asyncio.Semaphore = request.app.state.semaphore
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict[str, Any] | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        streaming = Summarizer(StreamingBackend(base.backend, emit, streamer_factory), base.cfg, base.data_cfg)

        def work() -> None:
            request_id_var.set(rid)
            t = time.perf_counter()
            try:
                r = streaming.summarize(req.text, length=req.length)
                latency = (time.perf_counter() - t) * 1000
                logger.info("stream length=%s strategy=%s new_tokens=%d latency_ms=%.0f", r.length, r.strategy,
                            r.new_tokens, latency)
                emit({"type": "done", "request_id": rid, "summary": r.summary, "length": r.length,
                      "strategy": r.strategy, "latency_ms": round(latency, 1), "new_tokens": r.new_tokens,
                      "generate_calls": r.generate_calls, "decoding": "greedy",
                      "backend": _backend_info(base.backend).model_dump()})
            except Exception as err:
                logger.exception("stream lỗi")
                emit({"type": "error", "request_id": rid, "detail": f"{type(err).__name__}: {err}"})
            finally:
                emit(None)

        async def events():
            async with semaphore:
                task = asyncio.ensure_future(run_in_threadpool(work))
                yield sse_event("start", {"request_id": rid, "length": req.length,
                                          "backend": _backend_info(base.backend).model_dump()})
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    kind = event.pop("type")
                    yield sse_event(kind, event)
                await task

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
