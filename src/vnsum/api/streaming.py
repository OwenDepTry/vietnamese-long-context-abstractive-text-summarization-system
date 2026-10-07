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


def default_streamer_factory(tokenizer: Any):
    from transformers import TextIteratorStreamer

    return TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Một event Server-Sent Events (``event:`` + ``data:`` JSON một dòng)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class StreamingBackend:
    """Backend bọc ngoài: cùng interface ``generate(texts, gen_kwargs)`` như backend của phase 4."""

    def __init__(self, base: Any, emit: Emit, streamer_factory: Callable[[Any], Any] | None = None) -> None:
        self.base = base
        self.emit = emit
        self.streamer_factory = streamer_factory or default_streamer_factory
        self.name, self.device, self.dtype = base.name, base.device, base.dtype
        self.tokenizer = base.tokenizer
        self.calls = 0

    def generate(self, texts: Sequence[str], gen_kwargs: dict[str, Any]):
        if len(texts) != 1:
            raise ValueError("Streaming chỉ hỗ trợ batch 1")
        call = self.calls
        self.calls += 1
        streamer = self.streamer_factory(self.tokenizer)
        kwargs = {k: v for k, v in gen_kwargs.items() if k != "early_stopping"}
        kwargs.update(num_beams=1, streamer=streamer)   # streamer của transformers không hỗ trợ beam search
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
