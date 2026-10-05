"""Client gọi LLM qua HTTP (requests): Anthropic Messages API hoặc API kiểu OpenAI Chat Completions.

Khóa API chỉ đọc từ biến môi trường ``api_key_env`` trong config.
Mọi lời gọi phải đi qua ``ApiGate``: không có ``--allow_api`` thì không gửi request nào.
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ApiNotAllowedError(RuntimeError):
    """Gọi API trả phí khi chưa bật --allow_api."""


@dataclass
class ApiGate:
    """Chốt chặn chi phí: đếm request dự kiến, chỉ cho gọi thật khi ``allowed``."""

    allowed: bool = False
    planned: dict[str, int] = field(default_factory=dict)
    sent: dict[str, int] = field(default_factory=dict)

    def plan(self, label: str, n: int) -> None:
        self.planned[label] = self.planned.get(label, 0) + int(n)

    def check(self, label: str) -> None:
        if not self.allowed:
            raise ApiNotAllowedError(f"{label}: cần --allow_api để gọi LLM API trả phí")

    def record(self, label: str, n: int = 1) -> None:
        self.sent[label] = self.sent.get(label, 0) + n


class LLMClient:
    def __init__(self, cfg: dict[str, Any], *, label: str, gate: ApiGate, session: Any = None) -> None:
        self.cfg = cfg
        self.label = label
        self.gate = gate
        self.provider = cfg["provider"]
        self.model = cfg["model"]
        self._session = session

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _api_key(self) -> str:
        key = os.environ.get(self.cfg["api_key_env"])
        if not key:
            raise RuntimeError(f"Biến môi trường {self.cfg['api_key_env']} chưa được đặt (khóa API cho {self.label}).")
        return key

    def _request(self, system: str | None, user: str) -> tuple[str, dict[str, Any]]:
        # temperature: null -> không gửi (một số model reasoning chỉ nhận giá trị mặc định).
        temperature = self.cfg.get("temperature", 0)
        max_tokens = int(self.cfg["max_tokens"])
        # OpenAI (model mới) dùng max_completion_tokens; Gemini/vLLM... dùng max_tokens.
        tokens_field = self.cfg.get("max_tokens_field", "max_tokens")
        timeout = float(self.cfg.get("timeout_seconds", 120))
        if self.provider == "anthropic":
            url = ANTHROPIC_URL
            headers = {"x-api-key": self._api_key(), "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user}],
            }
            if system:
                body["system"] = system
        else:
            url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {self._api_key()}", "content-type": "application/json"}
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            body = {"model": self.model, tokens_field: max_tokens, "messages": messages}
        if temperature is not None:
            body["temperature"] = float(temperature)

        body.update(self.cfg.get("extra_body") or {})  # tham số riêng của provider (vd. reasoning_effort)
        resp = self.session.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code != 200:
            raise _HttpError(resp.status_code, resp.text[:500])
        data = resp.json()
        self.last_raw = data
        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            usage = data.get("usage", {})
        else:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
        return text.strip(), usage

    def list_models(self) -> list[str]:
        """GET /models (miễn phí, không tốn token): ID model mà key này dùng được."""
        if self.provider == "anthropic":
            url, headers = "https://api.anthropic.com/v1/models?limit=1000", {
                "x-api-key": self._api_key(), "anthropic-version": ANTHROPIC_VERSION}
        else:
            url, headers = self.cfg["base_url"].rstrip("/") + "/models", {"Authorization": f"Bearer {self._api_key()}"}
        resp = self.session.get(url, headers=headers, timeout=float(self.cfg.get("timeout_seconds", 120)))
        if resp.status_code != 200:
            raise RuntimeError(f"{self.label}: HTTP {resp.status_code}: {resp.text[:300]}")
        return sorted(m["id"] for m in resp.json().get("data", []))

    def complete(self, user: str, system: str | None = None) -> tuple[str, dict[str, Any]]:
        """Một request, có retry với backoff cho lỗi tạm thời (429/5xx/timeout)."""
        self.gate.check(self.label)
        retries = int(self.cfg.get("max_retries", 5))
        for attempt in range(retries + 1):
            try:
                out = self._request(system, user)
                self.gate.record(self.label)
                return out
            except _HttpError as err:
                if err.status not in _RETRY_STATUS or attempt == retries:
                    raise RuntimeError(f"{self.label} ({self.model}) HTTP {err.status}: {err.body}") from None
            except Exception as err:  # noqa: BLE001 - chỉ retry lỗi mạng/timeout
                if attempt == retries or not _is_transient(err):
                    raise
            delay = min(60.0, 2.0**attempt) + random.random()
            logger.warning("%s: lỗi tạm thời, thử lại sau %.1fs (lần %d)", self.label, delay, attempt + 1)
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def map(self, items: Sequence[Any], fn: Callable[[Any], Any]) -> list[Any]:
        """Chạy ``fn`` trên từng item với ``concurrency`` luồng, giữ thứ tự."""
        workers = max(1, int(self.cfg.get("concurrency", 1)))
        if workers == 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))


def _is_transient(err: Exception) -> bool:
    try:
        import requests
    except ImportError:
        return False
    return isinstance(err, (requests.ConnectionError, requests.Timeout))


class _HttpError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body
