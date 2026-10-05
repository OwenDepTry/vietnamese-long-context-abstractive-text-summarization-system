"""Khoảng tin cậy bootstrap và so sánh cặp (paired bootstrap) giữa các hệ thống.

* CI của trung bình: percentile bootstrap, lấy lại mẫu có hoàn lại trên các mẫu test.
* So sánh cặp: mọi hệ thống chạy trên CÙNG mẫu, nên lấy lại cùng một bộ chỉ số mẫu cho
  hai hệ thống và tính hiệu trung bình -> CI của hiệu. p (hai phía) ước lượng bằng
  2 x min(P(hiệu ≤ 0), P(hiệu ≥ 0)) trên các lần lấy lại.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

METRICS = ("rouge1", "rouge2", "rougeL", "bertscore_f", "faithfulness")


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, confidence: float, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int, confidence: float, seed: int) -> dict[str, float]:
    """Hiệu trung bình a - b trên các cặp cùng mẫu, kèm CI và p hai phía."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a và b phải cùng kích thước (cùng các mẫu)")
    if a.size == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.quantile(boots, alpha)),
        "ci_high": float(np.quantile(boots, 1 - alpha)),
        "p_value": min(1.0, p),
        "n": int(d.size),
    }


def compute_significance(per_sample: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (ci_df, paired_df).

    ci_df: system, metric, n, mean, ci_low, ci_high (trên toàn bộ tập đánh giá).
    paired_df: reference, system, metric, n, diff, ci_low, ci_high, p_value — chỉ trên
    các mẫu mà cả hai hệ thống đều có giá trị (với faithfulness: các mẫu được judge chấm).
    """
    st = cfg.get("stats", {}) or {}
    n_boot = int(st.get("bootstrap_samples", 1000))
    conf = float(st.get("confidence", 0.95))
    seed = int(st.get("seed", cfg.get("seed", 42)))
    ref = st.get("reference_system", "vit5_lora")
    metrics = [m for m in METRICS if m in per_sample.columns]

    ci_rows = []
    for system, sdf in per_sample.groupby("system", sort=False):
        for m in metrics:
            vals = sdf[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot, conf, seed)
            ci_rows.append({"system": system, "metric": m, "n": int(vals.size), "mean": mean, "ci_low": lo, "ci_high": hi})

    paired_rows = []
    if ref in set(per_sample["system"]):
        wide = {m: per_sample.pivot_table(index="id", columns="system", values=m, aggfunc="first") for m in metrics}
        for system in per_sample["system"].unique():
            if system == ref:
                continue
            for m in metrics:
                w = wide[m]
                if ref not in w.columns or system not in w.columns:
                    continue
                both = w[[ref, system]].dropna()
                if both.empty:
                    continue
                res = paired_bootstrap(both[ref].to_numpy(), both[system].to_numpy(), n_boot, conf, seed)
                paired_rows.append({"reference": ref, "system": system, "metric": m, **res})
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)
