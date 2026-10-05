#!/usr/bin/env python
"""Kiểm tra cấu hình LLM API trong configs/eval.yaml trước khi chạy thật.

Mặc định (MIỄN PHÍ, không tốn token): với mỗi client trong mục ``llm``
(summarizer, judge), gọi GET /models để xem key có hợp lệ không và model
trong config có nằm trong danh sách model mà key dùng được không.

--probe (TỐN PHÍ, rất nhỏ): gửi đúng 1 request ngắn cho mỗi client với
ĐÚNG các tham số mà evaluate.py sẽ dùng (temperature, max_tokens, extra_body...),
để phát hiện lỗi tham số trước khi chạy hàng nghìn request.

    python scripts/check_llm_api.py --config configs/eval.yaml
    python scripts/check_llm_api.py --config configs/eval.yaml --probe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vnsum.eval.config import load_eval_config  # noqa: E402
from vnsum.eval.llm_client import ApiGate, LLMClient  # noqa: E402

_PROBE_TEXT = "Hà Nội hôm nay mưa to, nhiều tuyến phố ngập sâu khiến giao thông ùn tắc vào giờ tan tầm."


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--probe", action="store_true", help="gửi 1 request thật cho mỗi client (tốn phí rất nhỏ)")
    p.add_argument("--filter", default=None, help="chỉ in model ID chứa chuỗi này (vd. luna, sonnet)")
    args = p.parse_args(argv)

    cfg = load_eval_config(args.config)
    ok = True
    for name, ccfg in cfg["llm"].items():
        print(f"\n=== llm.{name}: {ccfg['provider']} / {ccfg['model']} (key: {ccfg['api_key_env']}) ===")
        client = LLMClient(ccfg, label=name, gate=ApiGate(allowed=args.probe))
        try:
            models = client.list_models()
        except Exception as err:  # noqa: BLE001
            print(f"  LỖI khi lấy danh sách model: {err}")
            ok = False
            continue
        shown = [m for m in models if not args.filter or args.filter.lower() in m.lower()]
        print(f"  Key hợp lệ. Key dùng được {len(models)} model" + (f"; khớp '{args.filter}':" if args.filter else ":"))
        for m in shown[:60]:
            print(f"    {m}")
        if len(shown) > 60:
            print(f"    ... và {len(shown) - 60} model khác (dùng --filter để lọc)")
        if ccfg["model"] in models:
            print(f"  OK: model '{ccfg['model']}' có trong danh sách.")
        else:
            print(f"  CẢNH BÁO: model '{ccfg['model']}' KHÔNG có trong danh sách -> sửa llm.{name}.model trong config.")
            ok = False
            continue

        if args.probe:
            prompt = f"Tóm tắt trong một câu ngắn:\n{_PROBE_TEXT}"
            try:
                text, usage = client.complete(prompt)
            except Exception as err:  # noqa: BLE001
                print(f"  PROBE LỖI: {err}")
                print("  Gợi ý: nếu lỗi nói về temperature -> đặt temperature: null; nếu về max_tokens -> "
                      "đặt max_tokens_field: max_completion_tokens; nếu về reasoning_effort -> bỏ extra_body.")
                ok = False
                continue
            print(f"  PROBE OK. Output: {text!r}")
            print(f"  Usage: {usage}")
            if not text.strip():
                print("  CẢNH BÁO: output rỗng — có thể token reasoning dùng hết max_tokens; tăng max_tokens "
                      "hoặc giảm reasoning_effort.")
                ok = False
    print("\nKẾT LUẬN:", "mọi client OK" if ok else "có vấn đề, xem các dòng LỖI/CẢNH BÁO ở trên")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
