"""Chọn ``target_modules`` cho LoRA từ tên module thật của model và đếm tham số.

Các hàm thuần (``summarize_module_names``, ``resolve_target_modules``) không cần
torch, để unit test chạy được offline.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# FFN của T5 gốc (relu): wi, wo. T5 v1.1 / gated-gelu: wi_0, wi_1, wo.
_FFN_CANDIDATES = ("wi", "wi_0", "wi_1", "wo")
_EXCLUDE = ("lm_head",)


def linear_module_names(model: Any) -> list[str]:
    """Tên đầy đủ của mọi ``nn.Linear`` trong model."""
    import torch.nn as nn

    return [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]


def summarize_module_names(names: Iterable[str]) -> dict[str, int]:
    """Đếm theo hậu tố (phần cuối tên): ``{'q': 36, 'k': 36, ..., 'wi': 24, 'lm_head': 1}``."""
    return dict(sorted(Counter(n.rsplit(".", 1)[-1] for n in names).items()))


def resolve_target_modules(names: Iterable[str], targets: list[str], include_ffn: bool) -> list[str]:
    """Kiểm tra hậu tố mục tiêu có thật trong model; thêm FFN nếu ``include_ffn``."""
    suffixes = summarize_module_names(names)
    chosen = list(dict.fromkeys(targets))
    if include_ffn:
        chosen += [s for s in _FFN_CANDIDATES if s in suffixes and s not in chosen]
    missing = [t for t in chosen if t not in suffixes]
    if missing:
        raise ValueError(
            f"lora.target_modules không có trong model: {missing}. Các hậu tố Linear hiện có: {sorted(suffixes)}"
        )
    bad = [t for t in chosen if t in _EXCLUDE]
    if bad:
        raise ValueError(f"Không gắn LoRA vào {bad} (output projection dùng chung vocab).")
    return chosen


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def report_modules(names: list[str], out_dir: str | Path) -> str:
    """In tóm tắt theo hậu tố, ghi danh sách đầy đủ ra file; trả về đường dẫn file."""
    summary = summarize_module_names(names)
    print(f"\n=== {len(names)} module nn.Linear trong model (đếm theo hậu tố) ===")
    for suffix, count in summary.items():
        example = next(n for n in names if n.rsplit(".", 1)[-1] == suffix)
        print(f"  {suffix:<10} x{count:<4} vd. {example}")
    path = Path(out_dir) / "linear_module_names.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"  (danh sách đầy đủ: {path})")
    return str(path)
