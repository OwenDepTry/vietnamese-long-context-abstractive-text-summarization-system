"""Token streaming cho /summarize/stream, không đổi logic của Summarizer (phase 4).

``StreamingBackend`` bọc backend thật (PyTorch hoặc ONNX): mỗi lần ``generate`` nó gắn một
``TextIteratorStreamer`` vào ``gen_kwargs``, chạy ``model.generate`` trong một luồng phụ và đẩy từng
đoạn văn bản ra qua ``emit``. Backend nào gọi ``model.generate(**gen_kwargs)`` của transformers
(GenerationMixin) đều stream được — cả ``ORTModelForSeq2SeqLM`` của optimum.

Giới hạn của transformers: streamer không dùng được với beam search, nên khi stream ta sinh bằng
greedy (``num_beams=1``); kết quả có thể khác /summarize (beam 4).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Sequence

Emit = Callable[[dict[str, Any] | None], None]


class StreamCancelled(RuntimeError):
    """Client đã đóng kết nối: dừng các lần sinh còn lại."""


def stop_criteria(stop: threading.Event):
    """``StoppingCriteriaList`` dừng ``generate`` khi ``stop`` được đặt; None nếu không có transformers."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:  # pragma: no cover - API luôn có transformers
        return None

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
            return stop.is_set()

    return StoppingCriteriaList([_Stop()])


def default_streamer_factory(tokenizer: Any):
    from transformers import TextIteratorStreamer

    return TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Một event Server-Sent Events (``event:`` + ``data:`` JSON một dòng)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class StreamingBackend:
    """Backend bọc ngoài: cùng interface ``generate(texts, gen_kwargs)`` như backend của phase 4."""

    max_batch_size = 1   # Summarizer đọc giá trị này: tóm tắt chunk từng cái một để stream theo từng lần sinh

    def __init__(self, base: Any, emit: Emit, streamer_factory: Callable[[Any], Any] | None = None,
                 stop: threading.Event | None = None) -> None:
        self.base = base
        self.emit = emit
        self.stop = stop or threading.Event()
        self.streamer_factory = streamer_factory or default_streamer_factory
        self.name, self.device, self.dtype = base.name, base.device, base.dtype
        self.tokenizer = base.tokenizer
        self.calls = 0

    def generate(self, texts: Sequence[str], gen_kwargs: dict[str, Any]):
        if len(texts) != 1:
            raise ValueError("Streaming chỉ hỗ trợ batch 1")
        if self.stop.is_set():   # văn bản dài: không sinh các chunk còn lại khi client đã đi
            raise StreamCancelled("client đã đóng kết nối")
        call = self.calls
        self.calls += 1
        streamer = self.streamer_factory(self.tokenizer)
        # early_stopping / length_penalty chỉ có nghĩa với beam search
        kwargs = {k: v for k, v in gen_kwargs.items() if k not in ("early_stopping", "length_penalty")}
        kwargs.update(num_beams=1, streamer=streamer)   # streamer của transformers không hỗ trợ beam search
        criteria = stop_criteria(self.stop)
        if criteria is not None:
            kwargs["stopping_criteria"] = criteria     # dừng giữa chừng khi client đóng kết nối
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["out"] = self.base.generate(list(texts), kwargs)
            except BaseException as err:  # chuyển lỗi về luồng gọi
                result["err"] = err
                streamer.end()            # để vòng lặp đọc streamer không bị treo

        worker = threading.Thread(target=run, name=f"generate-{call}", daemon=True)
        worker.start()
        for piece in streamer:
            if piece:
                self.emit({"type": "token", "call": call, "text": piece})
        worker.join()
        if "err" in result:
            raise result["err"]
        return result["out"]
