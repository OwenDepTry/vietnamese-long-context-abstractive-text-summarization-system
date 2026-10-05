"""Tổng hợp kết quả theo từng mẫu -> metrics.csv, report.md, qualitative.md.

Mọi con số trong report đều tính từ DataFrame kết quả theo từng mẫu (cũng được
lưu ra ``per_sample.csv``); không có số nào được nhập tay.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_COLUMNS = [
    "rouge1", "rouge2", "rougeL", "bertscore_p", "bertscore_r", "bertscore_f",
    "length_syllables",
]


def _aux_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("repetition_") or c.startswith("novel_")]


def _subsets(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    subsets = [("all", pd.Series(True, index=df.index))]
    for name in sorted(df["dataset"].unique()):
        subsets.append((f"dataset={name}", df["dataset"] == name))
    subsets.append(("long", df["is_long"].astype(bool)))
    subsets.append(("short", ~df["is_long"].astype(bool)))
    return subsets


def aggregate(per_sample: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Một dòng cho mỗi (system, subset): n, trung bình các chỉ số, thống kê faithfulness."""
    rows = []
    metrics = [c for c in METRIC_COLUMNS if c in per_sample.columns] + _aux_columns(per_sample)
    for system, sdf in per_sample.groupby("system", sort=False):
        for subset, mask in _subsets(sdf):
            part = sdf[mask]
            if part.empty:
                continue
            row: dict[str, Any] = {"system": system, "display_name": display_name(cfg or {}, system),
                                   "subset": subset, "n": len(part)}
            for m in metrics:
                vals = part[m].dropna()
                row[m] = float(vals.mean()) if len(vals) else math.nan
            if "faithfulness" in part.columns:
                faith = part["faithfulness"].dropna()
                row["faithfulness_n"] = int(len(faith))
                row["faithfulness_mean"] = float(faith.mean()) if len(faith) else math.nan
                row["faithfulness_pct_5"] = float(100 * (faith == 5).mean()) if len(faith) else math.nan
                row["faithfulness_pct_le2"] = float(100 * (faith <= 2).mean()) if len(faith) else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def display_name(cfg: dict[str, Any], system: str) -> str:
    """Tên hiển thị trong report: ``systems.<key>.display_name`` nếu có, ngược lại là key."""
    return str(cfg.get("systems", {}).get(system, {}).get("display_name") or system)


def _label(cfg: dict[str, Any], system: str) -> str:
    name = display_name(cfg, system)
    return f"**{name}**" if name != system else f"`{system}`"


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    return f"{v:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


LIMITATIONS = """\
- **ROUGE** đo trùng lặp bề mặt theo âm tiết (tokenizer giữ dấu, không tách từ, không stemming); không đo được diễn đạt lại hay tính đúng sự thật. Số liệu không so trực tiếp được với các bài báo dùng văn bản đã tách từ hoặc tokenizer khác.
- **Reference VietNews là phần tóm tắt (sapo) đi kèm bài báo**, thường bám sát các câu mở đầu, nên Lead-3 có thể được lợi trên VietNews; reference WikiLingua là các câu tóm tắt ý chính của từng phần hướng dẫn wikiHow.
- **BERTScore** dùng PhoBERT-base, lớp {layers}, input tách từ bằng {segmenter} (PhoBERT được huấn luyện với VnCoreNLP RDRSegmenter, nên có thể lệch tách từ), cắt ≤ {max_len} subword. **Không có baseline rescale cho PhoBERT → điểm là RAW**, thường dồn trong một dải hẹp ở mức cao; chỉ có ý nghĩa khi so tương đối giữa các hệ thống trong cùng bảng.
- **Faithfulness (LLM-as-Judge)**, khi được chạy, chỉ chấm tối đa {judge_n} mẫu/hệ thống (ngẫu nhiên, seed {judge_seed}) bằng một model ({judge_model}); {human_note} judge có thể sai và nhạy với cách viết prompt. Judge được chọn khác họ với LLM zero-shot để giảm thiên vị, nhưng không loại trừ hoàn toàn. Document dài hơn {judge_chars} ký tự bị cắt trước khi chấm.
- **Faithfulness ưu ái hệ thống chép nguyên văn:** Lead-3 chỉ chép câu từ văn bản gốc nên gần như luôn đạt điểm tối đa. Cần đọc faithfulness cùng tỷ lệ novel n-gram (mức độ diễn đạt lại) và ROUGE.
- **LLM zero-shot** nhận document gốc đầy đủ (≤ {llm_chars} ký tự) và sinh bằng API (không dùng được beam search / no_repeat_ngram), trong khi ViT5 nhận input đã lọc extractive ≤ 1024 token — hai điều kiện input khác nhau là có chủ đích (so sánh model ngữ cảnh dài với pipeline ngữ cảnh ngắn).
- **ViT5 công khai** (VietAI/vit5-base-vietnews-summarization) được fine-tune trên VietNews; không kiểm chứng được split train của nó có trùng tập test ở đây hay không. Trên WikiLingua nó là out-of-domain.
- Tập đánh giá là mẫu ngẫu nhiên (seed) của split test, thực tế gồm {per_ds}; nhóm "long" (document > {threshold} token) nhỏ hơn nhiều so với nhóm "short", nên chênh lệch ở nhóm long có độ bất định lớn hơn. {stats_note}
"""


def write_report(
    per_sample: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: dict[str, Any],
    out_path: Path,
    *,
    system_info: dict[str, str],
    skipped: dict[str, str],
    notes: list[str],
    reference_aux: dict[str, float] | None = None,
    significance: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> None:
    gen = cfg["generation"]
    allm = metrics[metrics["subset"] == "all"].set_index("system")
    aux = _aux_columns(per_sample)
    has_bs = "bertscore_f" in metrics.columns and metrics["bertscore_f"].notna().any()
    has_faith = "faithfulness_n" in metrics.columns and metrics["faithfulness_n"].fillna(0).sum() > 0

    lines = [
        "# Phase 3 — Báo cáo đánh giá",
        "",
        f"Sinh tự động lúc {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} bởi `scripts/evaluate.py`. "
        "Mọi con số dưới đây được tính từ `per_sample.csv` / `metrics.csv` trong cùng thư mục.",
        "",
        "## Thiết lập",
        "",
        f"- Tập đánh giá: split `{cfg['eval_set']['split']}`, "
        + ", ".join(f"{k}: {v:,} mẫu" for k, v in per_sample.drop_duplicates("id")["dataset"].value_counts().items())
        + f"; nhóm long (document > {cfg['eval_set']['long_doc_threshold_tokens']} token): "
        f"{int(per_sample.drop_duplicates('id')['is_long'].sum()):,} mẫu.",
        f"- Generation (model seq2seq): num_beams={gen['num_beams']}, max_new_tokens={gen['max_new_tokens']}, "
        f"no_repeat_ngram_size={gen['no_repeat_ngram_size']}, early_stopping={gen.get('early_stopping', True)}.",
        "- Hệ thống: " + "; ".join(f"{_label(cfg, k)} = {v}" for k, v in system_info.items()) + ".",
        "- ROUGE: F1 trung bình theo mẫu, tokenizer âm tiết giữ dấu tiếng Việt (thang 0–100).",
    ]
    if has_bs:
        bs = cfg["bertscore"]
        lines.append(f"- BERTScore: {bs['model']}, num_layers={bs['num_layers']}, tách từ {bs['word_segmentation']}, "
                     f"≤ {bs['max_length']} subword, **raw (không rescale)**, thang 0–100.")
    if has_faith:
        judged = allm["faithfulness_n"].dropna().astype(int)
        jt = cfg["llm"]["judge"].get("temperature", 0)
        jt_text = f"temperature {jt}" if jt is not None else "temperature mặc định của model (model không cho đặt)"
        lines.append(f"- Faithfulness: LLM-as-Judge `{cfg['llm']['judge']['model']}`, rubric 1–5, {jt_text}, chấm trên cùng một mẫu ngẫu nhiên (seed {cfg['judge']['seed']}): "
                     + ", ".join(f"`{k}` {v:,} mẫu" for k, v in judged.items() if v > 0) + ".")
    for note in notes:
        lines.append(f"- {note}")
    if skipped:
        lines += ["", "**Không có trong kết quả:** " + "; ".join(f"{_label(cfg, k.split(':')[-1]) if ':' in k else _label(cfg, k)}{' (judge)' if k.startswith('judge:') else ''} ({v})" for k, v in skipped.items()) + "."]

    # Bảng chính
    headers = ["Hệ thống", "n", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    if has_bs:
        headers.append("BERTScore-F (raw)")
    if has_faith:
        headers += ["Faithfulness (1–5)", "n chấm"]
    rows = []
    for system, r in allm.iterrows():
        row = [_label(cfg, system), _fmt(int(r["n"])), _fmt(r["rouge1"]), _fmt(r["rouge2"]), _fmt(r["rougeL"])]
        if has_bs:
            row.append(_fmt(r.get("bertscore_f")))
        if has_faith:
            n_j = r.get("faithfulness_n")
            row += [_fmt(r.get("faithfulness_mean")), _fmt(int(n_j)) if pd.notna(n_j) else "—"]
        rows.append(row)
    lines += ["", "## Kết quả chính (toàn bộ tập đánh giá)", "", _table(headers, rows)]

    # Theo dataset và long/short
    for title, prefix in (("Theo dataset", "dataset="), ("Theo độ dài document (long vs short)", None)):
        sub = metrics[metrics["subset"].str.startswith("dataset=")] if prefix else metrics[metrics["subset"].isin(["long", "short"])]
        hdr = ["Hệ thống", "Tập con", "n", "ROUGE-1", "ROUGE-2", "ROUGE-L"] + (["BERTScore-F"] if has_bs else [])
        body = []
        for _, r in sub.iterrows():
            row = [_label(cfg, r["system"]), r["subset"].replace("dataset=", ""), _fmt(int(r["n"])),
                   _fmt(r["rouge1"]), _fmt(r["rouge2"]), _fmt(r["rougeL"])]
            if has_bs:
                row.append(_fmt(r.get("bertscore_f")))
            body.append(row)
        lines += ["", f"## {title}", "", _table(hdr, body)]

    # Chỉ số phụ
    hdr = ["", "Độ dài TB (âm tiết)"] + [c.replace("_", " ") + " (%)" for c in aux]
    body = []
    for system, r in allm.iterrows():
        body.append([_label(cfg, system), _fmt(r["length_syllables"], 1)] + [_fmt(r[c]) for c in aux])
    if reference_aux:
        body.append(["*reference (tham chiếu)*", _fmt(reference_aux.get("length_syllables"), 1)]
                    + [_fmt(reference_aux.get(c)) for c in aux])
    lines += ["", "## Chỉ số phụ (toàn bộ tập đánh giá)", "",
              "Lặp = tỷ lệ n-gram trong summary là bản lặp của n-gram đứng trước. Novel = tỷ lệ n-gram của summary không có trong document gốc.",
              "", _table(hdr, body)]

    # Phân bố điểm faithfulness
    if has_faith:
        body = []
        for system, sdf in per_sample.groupby("system", sort=False):
            faith = sdf["faithfulness"].dropna()
            if faith.empty:
                continue
            counts = [int((faith == s).sum()) for s in (1, 2, 3, 4, 5)]
            errors = int(sdf.get("judge_error", pd.Series(dtype=object)).notna().sum())
            body.append([_label(cfg, system)] + [_fmt(c) for c in counts] + [_fmt(errors), _fmt(float(faith.mean()))])
        lines += ["", "## Phân bố điểm faithfulness", "",
                  _table(["Hệ thống", "1", "2", "3", "4", "5", "lỗi parse", "TB"], body)]

    if significance is not None:
        lines += _significance_section(cfg, *significance)

    bs = cfg["bertscore"]
    lines += ["", "## Giới hạn của phương pháp đánh giá", "", LIMITATIONS.format(
        layers=bs["num_layers"], segmenter=bs["word_segmentation"], max_len=bs["max_length"],
        judge_n=cfg["judge"]["sample_size"], judge_seed=cfg["judge"]["seed"], judge_model=cfg["llm"]["judge"]["model"],
        judge_chars=cfg["judge"]["max_document_chars"],
        llm_chars=cfg["systems"].get("llm_zeroshot", {}).get("max_input_chars", "—"),
        per_ds=", ".join(f"{k} {v:,} mẫu" for k, v in per_sample.drop_duplicates("id")["dataset"].value_counts().items()),
        threshold=cfg["eval_set"]["long_doc_threshold_tokens"],
        stats_note=("Khoảng tin cậy và so sánh cặp ở trên dùng bootstrap trên toàn bộ tập, chưa tách theo dataset hay độ dài; "
                    "các bảng theo tập con chỉ là trung bình điểm."
                    if significance is not None else "Report này chưa có khoảng tin cậy hay kiểm định ý nghĩa thống kê."),
        human_note=("độ tin cậy của judge được đối chiếu với người chấm trong `human_eval/agreement.md` (nếu đã chạy);"
                    if cfg.get("human_eval") else "không có đối chứng người chấm;"),
    )]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_METRIC_LABEL = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
                 "bertscore_f": "BERTScore-F", "faithfulness": "Faithfulness"}


def _significance_section(cfg: dict[str, Any], ci_df: pd.DataFrame, paired_df: pd.DataFrame) -> list[str]:
    st = cfg.get("stats", {}) or {}
    n_boot = int(st.get("bootstrap_samples", 1000))
    conf = float(st.get("confidence", 0.95))
    pct = f"{conf * 100:.0f}%"
    lines: list[str] = []
    if not ci_df.empty:
        metrics = [m for m in _METRIC_LABEL if m in set(ci_df["metric"])]
        body = []
        for system in ci_df["system"].unique():
            row = [_label(cfg, system)]
            for m in metrics:
                r = ci_df[(ci_df.system == system) & (ci_df.metric == m)]
                row.append("—" if r.empty else f"{r.iloc[0]['mean']:.2f} [{r.iloc[0]['ci_low']:.2f}, {r.iloc[0]['ci_high']:.2f}]")
            body.append(row)
        lines += ["", f"## Khoảng tin cậy {pct} (bootstrap, {n_boot:,} lần lấy lại)", "",
                  "Định dạng: trung bình [cận dưới, cận trên]. Faithfulness tính trên các mẫu được judge chấm.", "",
                  _table(["Hệ thống"] + [_METRIC_LABEL[m] for m in metrics], body)]
    if not paired_df.empty:
        ref = paired_df["reference"].iloc[0]
        body = []
        for _, r in paired_df.iterrows():
            sig = r["ci_low"] > 0 or r["ci_high"] < 0
            verdict = ("có ý nghĩa: " + (f"{display_name(cfg, ref)} cao hơn" if r["diff"] > 0 else f"{display_name(cfg, r['system'])} cao hơn")
                       if sig else "chưa đủ bằng chứng khác biệt")
            body.append([_label(cfg, r["system"]), _METRIC_LABEL[r["metric"]], _fmt(int(r["n"])), f"{r['diff']:+.2f}",
                         f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]",
                         f"< {1 / n_boot:.3f}" if r["p_value"] == 0 else f"{r['p_value']:.3f}", verdict])
        lines += ["", f"## So sánh cặp với {_label(cfg, ref)} (paired bootstrap)", "",
                  f"Hiệu = {display_name(cfg, ref)} − hệ thống, tính trên cùng các mẫu. Khác biệt được coi là có ý nghĩa khi "
                  f"CI {pct} của hiệu không chứa 0. Chưa hiệu chỉnh cho nhiều phép so sánh (multiple comparisons), "
                  "nên các p sát ngưỡng cần đọc thận trọng.", "",
                  _table(["So với", "Chỉ số", "n", "Hiệu", f"CI {pct}", "p", "Kết luận"], body)]
    return lines


def write_qualitative(per_sample: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict[str, Any], out_path: Path) -> None:
    q = cfg["qualitative"]
    system, k, doc_chars = q["system"], int(q["k"]), int(q["document_chars"])
    lines = [f"# Phân tích định tính — {_label(cfg, system)}", ""]
    sdf = per_sample[per_sample["system"] == system]
    judged = sdf.dropna(subset=["faithfulness"]) if "faithfulness" in sdf.columns else sdf.iloc[0:0]
    if judged.empty:
        lines.append("Chưa có điểm faithfulness cho hệ thống này (judge chưa chạy, ví dụ do `--skip_judge` hoặc thiếu `--allow_api`). "
                     "Chạy lại với judge để có 5 mẫu tốt nhất / tệ nhất.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    docs = eval_df.set_index(eval_df["id"].astype(str))
    ordered = judged.sort_values(["faithfulness", "rougeL", "id"], ascending=[False, False, True])
    best = ordered.head(k)
    worst = judged.sort_values(["faithfulness", "rougeL", "id"], ascending=[True, True, True]).head(k)
    lines.append(f"Chọn từ {len(judged)} mẫu đã chấm faithfulness. Document được rút gọn còn {doc_chars} ký tự đầu.")
    for title, part in ((f"{k} mẫu tốt nhất (faithfulness cao nhất)", best), (f"{k} mẫu tệ nhất (faithfulness thấp nhất)", worst)):
        lines += ["", f"## {title}"]
        for n, (_, r) in enumerate(part.iterrows(), 1):
            src = docs.loc[str(r["id"])]
            doc = str(src["document"])
            doc_short = doc[:doc_chars] + (" …" if len(doc) > doc_chars else "")
            claims = r.get("unsupported_claims") or []
            claims_md = "\n".join(f"  - {c}" for c in claims) if len(claims) else "  - (không có)"
            lines += [
                "",
                f"### {n}. `{r['id']}` ({r['dataset']}, {'long' if r['is_long'] else 'short'}) — faithfulness {int(r['faithfulness'])}/5, ROUGE-L {r['rougeL']:.2f}",
                "",
                f"**Document (rút gọn):** {doc_short}",
                "",
                f"**Reference:** {src['summary']}",
                "",
                f"**Prediction:** {r['prediction']}",
                "",
                "**Lỗi cụ thể (unsupported claims theo judge):**",
                claims_md,
                "",
                f"**Nhận xét của judge:** {r.get('judge_rationale') or '—'}",
            ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
