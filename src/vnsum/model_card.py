"""Model card (README.md trên Hugging Face Hub) sinh từ kết quả đánh giá trong ``results/``.

Số liệu lấy từ ``results/metrics.csv`` (phase 3) để model card không lệch với báo cáo; phần còn lại là mô tả
cố định của dự án. Không phụ thuộc torch / huggingface_hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

GITHUB_URL = "https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system"
BASE_MODEL = "VietAI/vit5-base"

SYSTEM_LABELS = {
    "vit5_lora": "**Model này** (ViT5-base + LoRA)",
    "vit5_public": "VietAI/vit5-base-vietnews-summarization",
    "llm_zeroshot": "LLM zero-shot",
    "lead3": "Lead-3 (3 câu đầu)",
}
SUBSET_LABELS = {"all": "Toàn bộ", "dataset=vietnews": "VietNews", "dataset=wikilingua_vi": "WikiLingua-vi",
                 "long": "Văn bản > 1024 token"}


def metrics_table(metrics: pd.DataFrame) -> tuple[str, int]:
    """Bảng ROUGE / BERTScore / faithfulness theo hệ thống (tập test đầy đủ) và theo tập con cho model này."""
    rows = ["| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F (raw) | Faithfulness (1–5) |", "|---|---|---|---|---|---|"]
    allm = metrics[metrics["subset"] == "all"].set_index("system")
    for sys_name, label in SYSTEM_LABELS.items():
        if sys_name not in allm.index:
            continue
        r = allm.loc[sys_name]
        faith = f"{r['faithfulness_mean']:.2f}" if pd.notna(r.get("faithfulness_mean")) else "—"
        rows.append(f"| {label} | {r.rouge1:.2f} | {r.rouge2:.2f} | {r.rougeL:.2f} | {r.bertscore_f:.2f} | {faith} |")
    n_all = int(allm.loc["vit5_lora", "n"]) if "vit5_lora" in allm.index else 0
    sub = metrics[metrics["system"] == "vit5_lora"].set_index("subset")
    rows += ["", "Model này theo tập con:", "", "| Tập con | n | ROUGE-1 | ROUGE-2 | ROUGE-L | Độ dài TB (âm tiết) |",
             "|---|---|---|---|---|---|"]
    for key, label in SUBSET_LABELS.items():
        if key in sub.index:
            r = sub.loc[key]
            rows.append(f"| {label} | {int(r.n):,} | {r.rouge1:.2f} | {r.rouge2:.2f} | {r.rougeL:.2f} | {r.length_syllables:.1f} |")
    return "\n".join(rows), n_all


def build_model_card(metrics: pd.DataFrame, repo_id: str, *, has_onnx: bool = False,
                     extra: dict[str, Any] | None = None) -> str:
    table, n_all = metrics_table(metrics)
    onnx_note = ("\n- `onnx/int8/`: bản ONNX INT8 (CPU) xuất bằng `scripts/export_onnx.py`, dùng với "
                 "`optimum.onnxruntime.ORTModelForSeq2SeqLM`." if has_onnx else "")
    extra_lines = "".join(f"\n- {k}: {v}" for k, v in (extra or {}).items())
    return f"""---
language:
- vi
license: other
library_name: transformers
pipeline_tag: summarization
base_model: {BASE_MODEL}
tags:
- summarization
- vietnamese
- t5
- lora
datasets:
- vietnews
- wiki_lingua
---

# ViT5-base + LoRA: tóm tắt văn bản tiếng Việt

Model tóm tắt trừu tượng (abstractive) tiếng Việt: [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) fine-tune
bằng LoRA (r=16, `q,k,v,o`) rồi merge vào trọng số. Mã nguồn, dữ liệu và toàn bộ báo cáo đánh giá:
[GitHub]({GITHUB_URL}).

## Nội dung repo

- Trọng số đã merge (dùng như một model T5 bình thường, không cần `peft`).{onnx_note}{extra_lines}

## Cách dùng

Model được train trên văn bản đã **chuẩn hóa** và, với văn bản dài hơn 1,024 token, đã **lọc câu** (BM25). Cách
đúng nhất là dùng pipeline của dự án (chuẩn hóa, chọn chiến lược, tóm tắt phân cấp cho văn bản dài):

```python
# trong thư mục repo GitHub, sau khi cài requirements
from huggingface_hub import snapshot_download
from vnsum.inference.config import load_inference_config
from vnsum.inference.predictor import Summarizer

model_dir = snapshot_download("{repo_id}", ignore_patterns=["onnx/*"])
cfg = load_inference_config("configs/inference.yaml")
summ = Summarizer.from_config(cfg, "pytorch", model_path=model_dir)
print(summ.summarize(text, length="medium").summary)   # short | medium | long
```

Với văn bản ngắn (≤ 1,024 token) có thể gọi thẳng `transformers`:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSeq2SeqLM.from_pretrained("{repo_id}")
ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
out = model.generate(ids, num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, early_stopping=True)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Train

- Dữ liệu: 116,557 mẫu train (VietNews 99,133 + WikiLingua-vi 17,424), input lọc câu `extractive_filter[bm25]` ≤ 1,024 token.
- LoRA r=16, alpha=32 trên `q,k,v,o`; bf16; batch hiệu dụng 32; LR 5e-4; 1 epoch (3,643 step) trên RTX 4050 Laptop 6GB (7 giờ 26 phút).

## Đánh giá

{n_all:,} mẫu test (VietNews + WikiLingua-vi), beam 4, `max_new_tokens` 128, `no_repeat_ngram_size` 3. ROUGE F1 với
tokenizer âm tiết **giữ dấu** (không so trực tiếp được với bài báo dùng văn bản đã tách từ). BERTScore dùng PhoBERT,
điểm raw. Faithfulness do LLM judge chấm trên 200 mẫu/hệ thống; đối chiếu với một người chấm trên 50 mục cho kappa
trọng số 0.70.

{table}

## Giới hạn

- **Riêng VietNews, model thua** `VietAI/vit5-base-vietnews-summarization` (ROUGE-L 26.75 so với 32.11); chỉ hơn trên
  tập hỗn hợp nhờ WikiLingua. Không kiểm chứng được model công khai có train trên tập test VietNews hay không.
- Tóm tắt **ngắn hơn** reference (~27 so với ~39 âm tiết) và chủ yếu trích câu (novel 2-gram ~18% so với ~54%).
- Có thể **sai sự thật** (faithfulness trung bình 4.01/5): đã gặp gán nhầm số liệu, ngày tháng, chức danh. Không dùng
  cho việc cần độ chính xác cao mà không kiểm tra lại.
- Ngoài miền dữ liệu train (tin tức, hướng dẫn wikiHow), ví dụ văn bản hành chính, model thường chỉ chép câu mở đầu.

## Giấy phép

`license: other`. Trọng số kế thừa điều khoản của `{BASE_MODEL}` và của dữ liệu train: WikiLingua theo CC BY 3.0
(nội dung wikiHow); VietNews theo bài báo VNDS (Nguyen et al., NICS 2019) và repo `ThanhChinhBK/vietnews`.
Kiểm tra các điều khoản này trước khi dùng cho mục đích thương mại.
"""


def load_metrics(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")
