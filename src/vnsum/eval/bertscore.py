"""BERTScore với PhoBERT (vinai/phobert-base), cài đặt trực tiếp.

Không dùng thư viện ``bert-score`` (bản cuối 0.3.13, 02/2023, trước transformers 5):
tokenizer PhoBERT không khai báo ``model_max_length`` nên thư viện không cắt input
và vượt 258 vị trí của PhoBERT. Cài đặt ở đây theo đúng thuật toán của thư viện
(Zhang et al., 2020; hàm ``greedy_cos_idf`` với idf tắt):

* embedding = hidden state của lớp ``num_layers`` (đếm từ 1, như thư viện cắt bớt lớp);
* chuẩn hóa L2, ghép tham lam theo cosine; bỏ token đặc biệt ``<s>``/``</s>``
  (thư viện gán trọng số 0 cho chúng);
* P = trung bình max cosine theo token của prediction, R = theo token của reference, F = 2PR/(P+R);
* KHÔNG rescale theo baseline (không có baseline cho PhoBERT) -> điểm RAW.

PhoBERT yêu cầu input đã tách từ ("Hà_Nội"); mỗi chuỗi bị cắt còn ≤ ``max_length`` subword.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def greedy_match(hyp_emb, hyp_mask, ref_emb, ref_mask):
    """hyp_emb: B×H×D, ref_emb: B×R×D (đã chuẩn hóa L2); mask: B×H / B×R (1 = token được tính).

    Trả (P, R, F) dạng tensor B. Mẫu có prediction hoặc reference rỗng -> 0.
    """
    import torch

    sim = torch.bmm(hyp_emb, ref_emb.transpose(1, 2))  # B×H×R
    pair_mask = hyp_mask.unsqueeze(2).bool() & ref_mask.unsqueeze(1).bool()
    sim = sim.masked_fill(~pair_mask, float("-inf"))
    best_for_hyp = sim.max(dim=2).values  # B×H
    best_for_ref = sim.max(dim=1).values  # B×R
    hm, rm = hyp_mask.float(), ref_mask.float()
    best_for_hyp = torch.where(hyp_mask.bool(), best_for_hyp, torch.zeros_like(best_for_hyp))
    best_for_ref = torch.where(ref_mask.bool(), best_for_ref, torch.zeros_like(best_for_ref))
    p = best_for_hyp.sum(1) / hm.sum(1).clamp(min=1)
    r = best_for_ref.sum(1) / rm.sum(1).clamp(min=1)
    empty = (hm.sum(1) == 0) | (rm.sum(1) == 0)
    p = p.masked_fill(empty, 0.0)
    r = r.masked_fill(empty, 0.0)
    f = torch.where((p + r) > 0, 2 * p * r / (p + r), torch.zeros_like(p))
    return p, r, f


class PhoBertScorer:
    def __init__(self, cfg: dict[str, Any], device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from vnsum.data.preprocess import word_segment

        self.cfg = cfg
        self.num_layers = int(cfg["num_layers"])
        self.max_length = int(cfg["max_length"])
        self.batch_size = int(cfg.get("batch_size", 32))
        self.segment_backend = cfg.get("word_segmentation", "underthesea")
        self._word_segment = word_segment
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
        self.model = AutoModel.from_pretrained(cfg["model"]).to(self.device).eval()
        n_layers = self.model.config.num_hidden_layers
        if not 1 <= self.num_layers <= n_layers:
            raise ValueError(f"bertscore.num_layers={self.num_layers} ngoài khoảng 1..{n_layers}")
        self.truncated = 0
        self.total = 0

    def segment(self, texts: Sequence[str]) -> list[str]:
        return [self._word_segment(t, self.segment_backend) if t.strip() else "" for t in texts]

    def _embed(self, texts: Sequence[str]):
        import torch

        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        full = self.tokenizer(list(texts), add_special_tokens=True)["input_ids"]
        self.truncated += sum(1 for ids in full if len(ids) > self.max_length)
        self.total += len(texts)
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[self.num_layers]
        hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
        mask = enc["attention_mask"].clone()
        special = torch.zeros_like(mask, dtype=torch.bool)
        for tok_id in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id):
            if tok_id is not None:
                special |= enc["input_ids"] == tok_id
        mask = mask.masked_fill(special, 0)
        return hidden, mask

    def score(self, predictions: Sequence[str], references: Sequence[str]) -> list[dict[str, float]]:
        """Điểm P/R/F (0–100, raw) cho từng cặp; input là văn bản thường, tự tách từ."""
        if len(predictions) != len(references):
            raise ValueError("predictions và references phải cùng độ dài")
        hyps, refs = self.segment(predictions), self.segment(references)
        out: list[dict[str, float]] = []
        for start in range(0, len(hyps), self.batch_size):
            h_emb, h_mask = self._embed(hyps[start : start + self.batch_size])
            r_emb, r_mask = self._embed(refs[start : start + self.batch_size])
            p, r, f = greedy_match(h_emb, h_mask, r_emb, r_mask)
            for pi, ri, fi in zip(p.tolist(), r.tolist(), f.tolist()):
                out.append({"bertscore_p": 100 * pi, "bertscore_r": 100 * ri, "bertscore_f": 100 * fi})
        return out
