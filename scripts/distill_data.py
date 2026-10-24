#!/usr/bin/env python
"""Bước mở rộng B: tạo dữ liệu distillation từ LLM cho một phần tập train.

    # 1) Chỉ in ước tính số request / chi phí — KHÔNG gọi API
    python scripts/distill_data.py --config configs/distill.yaml
    # 2) Gọi API trả phí (mỗi kết quả ghi cache ngay; dừng giữa chừng chạy lại sẽ tiếp tục)
    python scripts/distill_data.py --config configs/distill.yaml --allow_api
    # Thử trước với vài bài:
    python scripts/distill_data.py --config configs/distill.yaml --limit 20 --allow_api

Kết quả: data/distill/train.parquet (id, dataset, input_text, summary = bản LLM) + stats.md.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from vnsum.data.distill import build_prompt, estimate_cost, filter_summaries, sample_train  # noqa: E402
from vnsum.eval.llm_client import ApiGate, LLMClient  # noqa: E402
from vnsum.eval.metrics import novel_ngram_rate, syllables  # noqa: E402
from vnsum.eval.systems import PredictionCache, fingerprint  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--allow_api", action="store_true", help="CHO PHÉP gọi LLM API trả phí")
    p.add_argument("--limit", type=int, default=None, help="chỉ xử lý N bài đầu của mẫu (thử nghiệm)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = cfg["llm"]

    df = sample_train(cfg["processed_dir"], cfg["samples"], int(cfg["seed"]))
    if args.limit:
        df = df.groupby("dataset", group_keys=False).head(max(1, args.limit // df["dataset"].nunique()))
    prompts = {i: build_prompt(llm["prompt"], t, int(cfg["max_input_chars"])) for i, t in zip(df["id"], df["input_text"])}

    fp = fingerprint({k: v for k, v in llm.items() if k not in ("concurrency", "timeout_seconds", "max_retries")}
                     | {"max_input_chars": cfg["max_input_chars"]})
    cache = PredictionCache(out_dir, "cache", fp)
    missing = cache.missing(list(df["id"]))
    est = estimate_cost([prompts[i] for i in missing], cfg["price"], int(cfg["price"]["expected_output_tokens"]))
    print(f"Mẫu: {len(df):,} bài {df['dataset'].value_counts().to_dict()}; đã có trong cache: {len(df) - len(missing):,}; "
          f"cần gọi: {len(missing):,}.")
    print(f"Ước tính THÔ cho phần cần gọi ({llm['provider']}/{llm['model']}): {est['requests']:,} request, "
          f"~{est['input_tokens'] / 1e6:.2f}M token input + ~{est['output_tokens'] / 1e6:.2f}M token output "
          f"≈ ${est['usd']:.2f} (theo đơn giá trong config; kiểm tra lại trang giá).")

    gate = ApiGate(allowed=args.allow_api)
    if missing:
        gate.plan("distill", len(missing))
        if not args.allow_api:
            print("CHƯA gọi API (thiếu --allow_api).")
        else:
            client = LLMClient(llm, label="distill", gate=gate)
            done = {"n": 0}

            def _one(i: str) -> None:
                text, _ = client.complete(prompts[i])
                cache.add({i: text})
                done["n"] += 1
                if done["n"] % 100 == 0:
                    print(f"  {done['n']:,}/{len(missing):,}", flush=True)

            client.map(missing, _one)
            print(f"Đã gửi {sum(gate.sent.values()):,} request.")

    have = df[df["id"].isin(cache.data)].copy()
    if have.empty:
        print("Chưa có bản tóm tắt nào trong cache; dừng.")
        return 0
    have["distill_summary"] = have["id"].map(cache.data)

    from vnsum.data.tokenization import load_tokenizer

    fcfg = cfg["filter"]
    tok = load_tokenizer(fcfg["tokenizer"], fcfg.get("token_env"))
    kept, dropped = filter_summaries(
        have, lambda texts: [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]],
        max_target_tokens=int(fcfg["max_target_tokens"]), min_syllables=int(fcfg["min_syllables"]))

    out = kept.rename(columns={"summary": "original_summary", "distill_summary": "summary"})
    out = out[["id", "dataset", "input_text", "summary", "original_summary", "distill_tokens"]]
    out.to_parquet(out_dir / "train.parquet", index=False)

    def _stats(col: str) -> tuple[float, float]:
        return (statistics.mean(len(syllables(s)) for s in out[col]),
                statistics.mean(novel_ngram_rate(s, d, 2) for s, d in zip(out[col], out["input_text"])))

    d_len, d_nov = _stats("summary")
    o_len, o_nov = _stats("original_summary")
    lines = [
        "# Dữ liệu distillation", "",
        f"Sinh lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `scripts/distill_data.py`; "
        f"model {llm['provider']}/{llm['model']}; chỉ từ split train.", "",
        f"- Có bản tóm tắt: {len(have):,} bài; giữ lại: **{len(out):,}** {out['dataset'].value_counts().to_dict()}.",
        f"- Bỏ: {dropped or 'không'}.", "",
        "| | Độ dài TB (âm tiết) | Novel 2-gram so với input (%) |", "|---|---|---|",
        f"| Bản tóm tắt LLM | {d_len:.1f} | {d_nov:.2f} |",
        f"| Bản tóm tắt gốc của dataset | {o_len:.1f} | {o_nov:.2f} |", "",
        "## Ví dụ", "",
    ]
    for r in out.head(3).itertuples():
        lines += [f"### {r.id}", "", f"- **Gốc:** {r.original_summary}", f"- **LLM:** {r.summary}", ""]
    (out_dir / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:11]))
    print(f"\nĐã ghi: {out_dir / 'train.parquet'}, {out_dir / 'stats.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
