"""Kiểm tra LLM-as-Judge bằng điểm người chấm (human evaluation).

Quy trình:
1. ``export``: lấy mẫu cố định (seed) các cặp (hệ thống, mẫu) ĐÃ được judge chấm, trộn thứ tự,
   ẩn tên hệ thống và điểm judge -> ``items.md`` (để đọc) + ``annotation_sheet.csv`` (để điền điểm)
   + ``key.csv`` (đáp án: item_id -> system, id, điểm judge; KHÔNG mở trước khi chấm xong).
2. Người chấm điền cột ``human_score`` (1–5, cùng rubric với judge).
3. ``analyze``: so điểm người với điểm judge -> ``agreement.md``.

Chỉ dùng numpy/pandas (không thêm dependency).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vnsum.eval.judge import RUBRIC

SCORES = (1, 2, 3, 4, 5)
SHEET_COLUMNS = ["item_id", "human_score", "human_notes"]


# --------------------------------------------------------------------------- export
def select_items(per_sample: pd.DataFrame, systems: list[str], n_items: int, seed: int) -> pd.DataFrame:
    """Chia đều n_items cho các hệ thống (phần dư cho hệ thống đứng đầu danh sách), chỉ lấy mẫu có điểm judge.

    Lấy ngẫu nhiên theo seed, không phụ thuộc thứ tự dòng đầu vào; thứ tự trình bày được trộn
    lẫn giữa các hệ thống để người chấm không đoán được hệ thống.
    """
    judged = per_sample.dropna(subset=["faithfulness"])
    judged = judged[judged["system"].isin(systems)]
    missing = [s for s in systems if s not in set(judged["system"])]
    if missing:
        raise ValueError(f"Không có mẫu đã được judge chấm cho: {missing}. Chạy evaluate.py (có judge) trước.")
    base, extra = divmod(int(n_items), len(systems))
    rng = random.Random(int(seed))
    picked = []
    for k, system in enumerate(systems):
        want = base + (1 if k < extra else 0)
        pool = sorted(judged.loc[judged["system"] == system, "id"].astype(str))
        rng_s = random.Random(f"{seed}:{system}")
        rng_s.shuffle(pool)
        if want > len(pool):
            raise ValueError(f"{system}: chỉ có {len(pool)} mẫu đã chấm, không đủ {want}")
        picked += [(system, i) for i in pool[:want]]
    rng.shuffle(picked)
    rows = []
    idx = judged.assign(id=judged["id"].astype(str)).set_index(["system", "id"])
    for n, (system, i) in enumerate(picked, 1):
        r = idx.loc[(system, i)]
        rows.append({"item_id": f"H{n:03d}", "system": system, "id": i, "dataset": r.get("dataset"),
                     "prediction": r["prediction"], "judge_score": int(r["faithfulness"]),
                     "judge_rationale": r.get("judge_rationale")})
    return pd.DataFrame(rows)


def export_items(items: pd.DataFrame, documents: pd.Series, out_dir: Path, max_document_chars: int,
                 *, force: bool = False) -> dict[str, Path]:
    """Ghi items.md, annotation_sheet.csv, key.csv. Không ghi đè bảng điểm đã có trừ khi force=True."""
    out_dir = Path(out_dir)
    sheet_path = out_dir / "annotation_sheet.csv"
    if sheet_path.exists() and not force:
        raise FileExistsError(f"{sheet_path} đã tồn tại (có thể chứa điểm bạn đã chấm). Dùng --force nếu muốn tạo lại.")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Chấm độ trung thực (faithfulness) — bản tóm tắt so với văn bản gốc",
        "",
        f"Có {len(items)} mục, đã trộn thứ tự và ẩn tên hệ thống. Với mỗi mục, đọc văn bản gốc rồi chấm bản "
        "tóm tắt theo thang dưới đây, điền điểm vào cột `human_score` của `annotation_sheet.csv` "
        "(ghi chú tùy chọn ở `human_notes`).",
        "",
        "Chỉ chấm việc bản tóm tắt có được văn bản gốc hỗ trợ hay không; KHÔNG chấm văn phong, độ đầy đủ hay độ dài. "
        "Đừng mở `key.csv` trước khi chấm xong.",
        "",
        "## Thang điểm (giống hệt rubric của LLM judge)",
        "",
        *[f"- {line}" for line in RUBRIC.splitlines()],
    ]
    for _, it in items.iterrows():
        doc = str(documents[it["id"]])
        truncated = len(doc) > int(max_document_chars)
        doc = doc[: int(max_document_chars)]
        lines += ["", "---", "", f"## {it['item_id']}", "", "**Văn bản gốc**"
                  + (f" (cắt còn {int(max_document_chars):,} ký tự đầu, như judge thấy)" if truncated else "") + ":", "",
                  *[f"> {p}" if p.strip() else ">" for p in doc.splitlines()], "",
                  "**Bản tóm tắt cần chấm:**", "", f"> {it['prediction']}", "",
                  f"Điểm của bạn ({it['item_id']}): ___"]
    paths = {"items": out_dir / "items.md", "sheet": sheet_path, "key": out_dir / "key.csv"}
    paths["items"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame({"item_id": items["item_id"], "human_score": "", "human_notes": ""}).to_csv(
        sheet_path, index=False, encoding="utf-8-sig")
    items[["item_id", "system", "id", "dataset", "judge_score", "judge_rationale"]].to_csv(
        paths["key"], index=False, encoding="utf-8-sig")
    return paths


# --------------------------------------------------------------------------- analyze
def read_sheet(path: Path) -> pd.DataFrame:
    """Đọc bảng điểm; tự nhận dấu phân cách ',' hoặc ';' (Excel ở một số locale lưu bằng ';')."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype={"item_id": str})
    df.columns = [str(c).strip() for c in df.columns]
    if "item_id" not in df.columns or "human_score" not in df.columns:
        raise ValueError(f"{path} phải có cột item_id và human_score (đang có {list(df.columns)})")
    score = pd.to_numeric(df["human_score"], errors="coerce")
    bad = df.loc[score.notna() & ~score.isin(SCORES), "item_id"].tolist()
    if bad:
        raise ValueError(f"human_score phải là số nguyên 1–5; sai ở: {bad}")
    df["human_score"] = score
    return df


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, labels=SCORES) -> float:
    """Cohen's kappa có trọng số bậc hai (thang thứ tự 1–5). Trả nan nếu không xác định."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    k = len(labels)
    pos = {v: i for i, v in enumerate(labels)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2
    denom = (w * exp).sum()
    return float("nan") if denom == 0 else float(1 - (w * obs).sum() / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman = Pearson trên hạng (hạng trung bình cho giá trị bằng nhau)."""
    ra = pd.Series(a, dtype=float).rank().to_numpy()
    rb = pd.Series(b, dtype=float).rank().to_numpy()
    if len(ra) < 2 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def agreement_stats(human: np.ndarray, judge: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    human, judge = np.asarray(human, dtype=int), np.asarray(judge, dtype=int)
    n = int(human.size)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    diff = human - judge
    out.update({
        "exact": float((diff == 0).mean()),
        "within1": float((np.abs(diff) <= 1).mean()),
        "mean_human": float(human.mean()),
        "mean_judge": float(judge.mean()),
        "mean_diff": float(diff.mean()),   # > 0: người chấm rộng tay hơn judge
        "spearman": spearman(human, judge),
        "qwk": quadratic_weighted_kappa(human, judge),
    })
    # CI bootstrap cho kappa (n nhỏ -> CI rộng; báo để không diễn giải quá mức).
    rng = np.random.default_rng(seed)
    boots = []
    for idx in rng.integers(0, n, size=(n_boot, n)):
        v = quadratic_weighted_kappa(human[idx], judge[idx])
        if not np.isnan(v):
            boots.append(v)
    if boots:
        out["qwk_ci"] = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    return out


def confusion(human: np.ndarray, judge: np.ndarray) -> pd.DataFrame:
    m = pd.crosstab(pd.Series(judge, name="judge"), pd.Series(human, name="human"))
    return m.reindex(index=list(SCORES), columns=list(SCORES), fill_value=0)


def _f(v: float, nd: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def analyze(sheet: pd.DataFrame, key: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[str, pd.DataFrame]:
    """Trả (nội dung agreement.md, bảng đã ghép). Chỉ tính trên các mục đã có human_score."""
    from vnsum.eval.report import display_name

    key = key.assign(item_id=key["item_id"].astype(str))
    merged = key.merge(sheet[[c for c in SHEET_COLUMNS if c in sheet.columns]], on="item_id", how="left")
    scored = merged.dropna(subset=["human_score"]).copy()
    scored["human_score"] = scored["human_score"].astype(int)
    scored["judge_score"] = scored["judge_score"].astype(int)
    name = (lambda s: display_name(cfg, s)) if cfg else (lambda s: s)

    st = agreement_stats(scored["human_score"].to_numpy(), scored["judge_score"].to_numpy())
    lines = ["# Đối chiếu LLM judge với người chấm", "",
             f"Đã chấm {len(scored)}/{len(merged)} mục. Mọi số dưới đây tính trên {len(scored)} mục đã chấm "
             "(một người chấm, rubric giống hệt judge, không biết tên hệ thống và điểm judge khi chấm).", ""]
    if len(scored) == 0:
        lines.append("Chưa có mục nào được chấm — điền cột `human_score` trong annotation_sheet.csv rồi chạy lại.")
        return "\n".join(lines) + "\n", merged
    ci = st.get("qwk_ci")
    lines += ["## Mức đồng thuận tổng", "",
              "| Chỉ số | Giá trị |", "|---|---|",
              f"| Trùng khớp tuyệt đối | {st['exact'] * 100:.1f}% |",
              f"| Lệch không quá 1 điểm | {st['within1'] * 100:.1f}% |",
              f"| Cohen's kappa trọng số bậc hai | {_f(st['qwk'], 3)}"
              + (f" (CI 95% bootstrap [{ci[0]:.3f}, {ci[1]:.3f}])" if ci else "") + " |",
              f"| Spearman | {_f(st['spearman'], 3)} |",
              f"| Điểm TB người / judge | {st['mean_human']:.2f} / {st['mean_judge']:.2f} |",
              f"| Hiệu TB (người − judge) | {st['mean_diff']:+.2f} |", ""]
    lines += ["## Theo hệ thống", "", "| Hệ thống | n | TB người | TB judge | Trùng khớp | Lệch ≤ 1 |", "|---|---:|---:|---:|---:|---:|"]
    for system, g in scored.groupby("system", sort=True):
        s = agreement_stats(g["human_score"].to_numpy(), g["judge_score"].to_numpy(), n_boot=0)
        lines.append(f"| {name(system)} | {s['n']} | {s['mean_human']:.2f} | {s['mean_judge']:.2f} | "
                     f"{s['exact'] * 100:.0f}% | {s['within1'] * 100:.0f}% |")
    cm = confusion(scored["human_score"].to_numpy(), scored["judge_score"].to_numpy())
    lines += ["", "## Ma trận nhầm lẫn (hàng = judge, cột = người)", "",
              "| judge \\ người | " + " | ".join(map(str, SCORES)) + " |", "|---" * (len(SCORES) + 1) + "|"]
    for j in SCORES:
        lines.append(f"| **{j}** | " + " | ".join(str(int(cm.loc[j, h])) for h in SCORES) + " |")
    big = scored[(scored["human_score"] - scored["judge_score"]).abs() >= 2].sort_values("item_id")
    lines += ["", f"## Các mục lệch từ 2 điểm trở lên ({len(big)})", ""]
    if big.empty:
        lines.append("Không có.")
    else:
        lines += ["| Mục | Hệ thống | Người | Judge | Lý do của judge | Ghi chú của người |", "|---|---|---:|---:|---|---|"]
        for _, r in big.iterrows():
            note = "" if pd.isna(r.get("human_notes")) else str(r.get("human_notes"))
            why = "" if pd.isna(r.get("judge_rationale")) else str(r.get("judge_rationale"))
            lines.append(f"| {r['item_id']} | {name(r['system'])} | {r['human_score']} | {r['judge_score']} | "
                         f"{why.replace('|', '/')} | {note.replace('|', '/')} |")
    lines += ["", "## Lưu ý", "",
              f"- Một người chấm, {len(scored)} mục: CI của kappa rộng; đây là kiểm tra độ tin cậy của judge, không phải "
              "nghiên cứu inter-annotator agreement đầy đủ.",
              *(["- Lead-3 không có trong mẫu: trích nguyên văn nên judge luôn chấm 5, không cho thông tin về độ tin cậy."]
                if "lead3" not in set(key["system"]) else []),
              "- Mẫu lấy đều theo hệ thống (không theo tỷ lệ điểm), nên % trùng khớp phụ thuộc phân bố điểm của từng hệ thống."]
    return "\n".join(lines) + "\n", merged
