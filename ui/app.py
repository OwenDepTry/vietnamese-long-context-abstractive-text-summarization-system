"""Streamlit UI cho vnsum: nhập văn bản -> 3 bản tóm tắt short / medium / long cạnh nhau.

UI chỉ gọi API qua HTTP (biến môi trường API_URL, mặc định http://localhost:8000); không nạp model.

    streamlit run ui/app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = float(os.environ.get("API_TIMEOUT_S", "600"))
LENGTHS = ("short", "medium", "long")
LABELS = {"short": "Ngắn", "medium": "Vừa", "long": "Dài"}


def get_health(client: httpx.Client) -> dict[str, Any] | None:
    try:
        r = client.get("/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


def summarize(client: httpx.Client, text: str, length: str) -> dict[str, Any]:
    """Gọi POST /summarize; trả JSON của API, hoặc {"error": ...} để UI hiển thị."""
    try:
        r = client.post("/summarize", json={"text": text, "length": length}, timeout=TIMEOUT_S)
    except httpx.HTTPError as err:
        return {"error": f"Không gọi được API ({type(err).__name__}): {err}"}
    if r.status_code == 200:
        return r.json()
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = r.text
    if isinstance(detail, list):  # lỗi validation của FastAPI
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return {"error": f"API trả {r.status_code}: {detail}"}


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="vnsum — tóm tắt tiếng Việt", layout="wide")
    st.title("Tóm tắt văn bản tiếng Việt")
    client = httpx.Client(base_url=API_URL)
    health = get_health(client)
    with st.sidebar:
        st.caption(f"API: {API_URL}")
        if health:
            b = health["backend"]
            st.success(f"API sẵn sàng — {b['name']} {b['dtype']} trên {b['device']}")
            st.caption("Số token tối đa: " + ", ".join(f"{LABELS.get(k, k)} {v}" for k, v in health["lengths"].items()))
        else:
            st.error("Không kết nối được API. Kiểm tra API đã chạy và biến API_URL.")
    max_chars = int(health["max_input_chars"]) if health else None
    text = st.text_area("Văn bản cần tóm tắt", height=280, max_chars=max_chars,
                        placeholder="Dán bài báo hoặc văn bản tiếng Việt vào đây...")
    if st.button("Tóm tắt", type="primary", disabled=not health or not text.strip()):
        cols = st.columns(len(LENGTHS))
        for col, length in zip(cols, LENGTHS):
            with col:
                st.subheader(LABELS[length])
                with st.spinner("Đang tóm tắt..."):
                    res = summarize(client, text, length)
                if "error" in res:
                    st.error(res["error"])
                    continue
                st.write(res["summary"])
                st.caption(f"{res['latency_ms']:,.0f} ms · {res['new_tokens']} token · chiến lược {res['strategy']}")


if __name__ == "__main__":
    main()
