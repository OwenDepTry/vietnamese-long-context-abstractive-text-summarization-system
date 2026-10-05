#!/usr/bin/env python
"""Phase 3: đánh giá và so sánh các hệ thống tóm tắt.

Ví dụ:
    # Smoke test CPU (không gọi API trả phí, không chấm judge) -> results/smoke/
    python scripts/evaluate.py --config configs/eval.yaml --limit 10 --skip_judge
    # Đầy đủ, KHÔNG gọi API: in ước tính số request cho LLM zero-shot + judge
    python scripts/evaluate.py --config configs/eval.yaml --model_path C:/vnsum-runs/vit5-vnsum-merged
    # Đầy đủ, CÓ gọi API trả phí (chỉ chạy khi đã đồng ý chi phí)
    python scripts/evaluate.py --config configs/eval.yaml --model_path ... --allow_api

Dự đoán và kết quả judge được cache trong results/: chạy lại chỉ sinh/chấm phần còn thiếu.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from vnsum.eval.config import load_eval_config  # noqa: E402
from vnsum.eval.dataset import load_eval_set  # noqa: E402
from vnsum.eval.judge import JudgeCache, judge_sample_ids, run_judge  # noqa: E402
from vnsum.eval.llm_client import ApiGate  # noqa: E402
from vnsum.eval.metrics import aux_per_sample, rouge_per_sample  # noqa: E402
from vnsum.eval.report import aggregate, write_qualitative, write_report  # noqa: E402
from vnsum.eval.stats import compute_significance  # noqa: E402
from vnsum.eval.systems import run_system  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=None, help="tổng số mẫu (smoke test); kết quả ghi vào <results_dir>/smoke")
    p.add_argument("--skip_judge", action="store_true", help="không chấm faithfulness")
    p.add_argument("--skip_bertscore", action="store_true")
    p.add_argument("--allow_api", action="store_true", help="CHO PHÉP gọi LLM API trả phí (LLM zero-shot + judge)")
    p.add_argument("--retry_judge_errors", action="store_true",
                   help="chấm lại các mục judge bị lỗi parse trong cache (vẫn cần --allow_api để thực sự gửi)")
    p.add_argument("--systems", default=None, help="danh sách hệ thống, phân cách bằng dấu phẩy (mặc định: mọi hệ thống enabled)")
    p.add_argument("--model_path", default=None, help="đường dẫn model ViT5 LoRA đã merge (override systems.vit5_lora.model_path)")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--processed_dir", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def resolve_lora_path(cfg: dict, sys_cfg: dict, override: str | None) -> str:
    if override:
        return override
    if sys_cfg.get("model_path"):
        return sys_cfg["model_path"]
    import yaml

    train_cfg_path = Path(cfg["paths"]["train_config"])
    with open(train_cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["paths"]["merged_dir"]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub.utils._http", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_eval_config(args.config)
    if args.processed_dir:
        cfg["paths"]["processed_dir"] = args.processed_dir
    results_dir = Path(args.results_dir or cfg["paths"]["results_dir"])
    if args.limit:
        results_dir = results_dir / "smoke"
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_df = load_eval_set(cfg, limit=args.limit)
    eval_df["id"] = eval_df["id"].astype(str)
    gate = ApiGate(allowed=args.allow_api)

    wanted = [s.strip() for s in args.systems.split(",")] if args.systems else None
    predictions: dict[str, dict[str, str]] = {}
    system_info: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name, sys_cfg in cfg["systems"].items():
        if not sys_cfg.get("enabled", True) or (wanted and name not in wanted):
            continue
        model_path = None
        if sys_cfg["type"] == "seq2seq":
            model_path = resolve_lora_path(cfg, sys_cfg, args.model_path) if name == "vit5_lora" else sys_cfg["model_path"]
            if name == "vit5_lora" and not os.path.isdir(model_path):
                skipped[name] = f"không thấy model đã merge ở {model_path}; truyền --model_path"
                print(f"[{name}] BỎ QUA — {skipped[name]}")
                continue
            desc = f"{sys_cfg['description']}, " if sys_cfg.get("description") else ""
            system_info[name] = f"{desc}{model_path} (input `{sys_cfg['input_column']}`)"
        elif sys_cfg["type"] == "llm":
            c = cfg["llm"][sys_cfg["client"]]
            system_info[name] = f"{c['provider']}/{c['model']} zero-shot (input `{sys_cfg['input_column']}`)"
        else:
            system_info[name] = f"{sys_cfg.get('num_sentences', 3)} câu đầu của `{sys_cfg['input_column']}`"
        preds = run_system(name, sys_cfg, eval_df, cfg, results_dir=results_dir, gate=gate, model_path=model_path)
        if preds is None:
            skipped[name] = "cần --allow_api"
            system_info.pop(name, None)
            continue
        predictions[name] = preds

    if not predictions:
        print("Không có hệ thống nào có dự đoán; dừng.")
        return 1

    # ----- ROUGE + chỉ số phụ theo từng mẫu -----
    aux_cfg = cfg["aux_metrics"]
    frames = []
    for name, preds in predictions.items():
        part = eval_df[["id", "dataset", "is_long", "document", "summary"]].copy()
        part.insert(0, "system", name)
        part["prediction"] = part["id"].map(preds).fillna("")
        rouge = pd.DataFrame(rouge_per_sample(part["prediction"].tolist(), part["summary"].tolist()), index=part.index)
        aux = pd.DataFrame(
            [aux_per_sample(p, d, int(aux_cfg["repetition_ngram"]), aux_cfg["novel_ngrams"])
             for p, d in zip(part["prediction"], part["document"])],
            index=part.index,
        )
        frames.append(pd.concat([part, rouge, aux], axis=1))
    per_sample = pd.concat(frames, ignore_index=True)
    ref_aux = pd.DataFrame(
        [aux_per_sample(s, d, int(aux_cfg["repetition_ngram"]), aux_cfg["novel_ngrams"])
         for s, d in zip(eval_df["summary"], eval_df["document"])]
    ).mean().to_dict()

    # ----- BERTScore -----
    notes: list[str] = []
    if cfg["bertscore"].get("enabled") and not args.skip_bertscore:
        from vnsum.eval.bertscore import PhoBertScorer

        scorer = PhoBertScorer(cfg["bertscore"])
        scores = scorer.score(per_sample["prediction"].tolist(), per_sample["summary"].tolist())
        per_sample = pd.concat([per_sample, pd.DataFrame(scores, index=per_sample.index)], axis=1)
        notes.append(f"BERTScore: {scorer.truncated:,}/{scorer.total:,} chuỗi (prediction + reference) bị cắt ở "
                     f"{cfg['bertscore']['max_length']} subword PhoBERT.")

    # ----- Faithfulness (LLM-as-Judge) -----
    judge_rows: list[dict] = []
    if cfg["judge"].get("enabled") and not args.skip_judge:
        jcfg = cfg["judge"]
        sample_ids = judge_sample_ids(eval_df["id"].tolist(), int(jcfg["sample_size"]), int(jcfg["seed"]))
        cache = JudgeCache(results_dir / "judge_cache.jsonl")
        docs = eval_df.set_index("id")["document"]
        for name in jcfg["systems"]:
            if name not in predictions:
                continue
            items = [{"id": i, "document": docs[i], "prediction": predictions[name][i]} for i in sample_ids]
            results = run_judge(name, items, cfg, cache=cache, gate=gate, retry_errors=args.retry_judge_errors)
            if results is None:
                skipped[f"judge:{name}"] = "cần --allow_api"
                continue
            for r in results:
                judge_rows.append({
                    "system": name, "id": r["id"], "faithfulness": r.get("score"),
                    "unsupported_claims": r.get("unsupported_claims") or [],
                    "judge_rationale": r.get("rationale"), "judge_error": r.get("error"),
                    "judge_recovered": bool(r.get("recovered")),
                })
    elif args.skip_judge:
        notes.append("Faithfulness: không chạy (--skip_judge).")
    judge_df = pd.DataFrame(judge_rows, columns=["system", "id", "faithfulness", "unsupported_claims", "judge_rationale",
                                                 "judge_error", "judge_recovered"])
    n_rec = int(judge_df["judge_recovered"].fillna(False).astype(bool).sum())
    if n_rec:
        by = judge_df[judge_df["judge_recovered"].fillna(False).astype(bool)].groupby("system").size().to_dict()
        notes.append(f"Faithfulness: {n_rec} kết quả judge có JSON không hợp lệ ({by}; chủ yếu do dấu \" không escape trong "
                     "unsupported_claims). Điểm được lấy từ trường `score` đứng đầu output, không gọi lại API; "
                     "danh sách unsupported_claims của các mục này giữ nguyên dạng thô.")
    per_sample = per_sample.merge(judge_df, on=["system", "id"], how="left")
    per_sample["faithfulness"] = pd.to_numeric(per_sample["faithfulness"], errors="coerce")

    # ----- Ghi kết quả -----
    metrics = aggregate(per_sample, cfg)
    ci_df, paired_df = compute_significance(per_sample, cfg)
    pd.concat([ci_df.assign(kind="ci"), paired_df.assign(kind="paired")], ignore_index=True).astype({"n": "Int64"}).to_csv(
        results_dir / "significance.csv", index=False, encoding="utf-8-sig")
    per_sample.drop(columns=["document"]).to_csv(results_dir / "per_sample.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(results_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    write_report(per_sample, metrics, cfg, results_dir / "report.md", system_info=system_info,
                 skipped=skipped, notes=notes, reference_aux=ref_aux,
                 significance=(ci_df, paired_df))
    write_qualitative(per_sample, eval_df, cfg, results_dir / "qualitative.md")

    show = metrics[metrics["subset"] == "all"].set_index("display_name")
    cols = [c for c in ("n", "rouge1", "rouge2", "rougeL", "bertscore_f", "faithfulness_mean") if c in show.columns]
    print("\n=== Kết quả (toàn bộ tập đánh giá) ===")
    print(show[cols].round(2).to_string())
    if gate.planned:
        total = sum(gate.planned.values())
        state = "đã gửi " + str(sum(gate.sent.values())) if gate.allowed else "CHƯA gửi (thiếu --allow_api)"
        print(f"\nLLM API: dự kiến {total} request {gate.planned} — {state}.")
    print(f"\nĐã ghi: {results_dir / 'metrics.csv'}, {results_dir / 'report.md'}, {results_dir / 'qualitative.md'}, {results_dir / 'per_sample.csv'}, {results_dir / 'significance.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
