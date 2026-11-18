# Phase 3 — Evaluation

← [Về README](../README.md)

```bash
# Smoke test (CPU, không gọi API trả phí) -> results/main/smoke/
python scripts/evaluate.py --config configs/eval.yaml --limit 10 --skip_judge --model_path <thư mục model đã merge>
# Đầy đủ, chưa gọi API: tính mọi chỉ số offline + in ước tính số request cho LLM zero-shot và judge
python scripts/evaluate.py --config configs/eval.yaml --model_path <thư mục model đã merge>
# Gọi API trả phí (khóa đọc từ biến môi trường, vd. ANTHROPIC_API_KEY)
python scripts/evaluate.py --config configs/eval.yaml --model_path <...> --allow_api
```

- **Hệ thống:** Lead-3, ViT5 + LoRA (phase 2), `VietAI/vit5-base-vietnews-summarization` (tham chiếu công khai), LLM zero-shot qua API. Các hệ thống chạy trên cùng mẫu test (seed), các model seq2seq dùng chung generation config.
- **ROUGE** dùng tokenizer âm tiết giữ dấu; tokenizer mặc định của `rouge-score` xóa ký tự ngoài ASCII ("Hà Nội" và "Hè Nổi" bị coi là giống nhau, có test chứng minh).
- **BERTScore** được cài đặt trực tiếp theo thuật toán gốc (greedy cosine, không idf), vì thư viện `bert-score` 0.3.13 không cắt input PhoBERT (tokenizer không khai báo `model_max_length`). Điểm là **raw**, vì không có baseline rescale cho PhoBERT.
- **API trả phí** chỉ được gọi khi có `--allow_api`. Dự đoán và kết quả judge được ghi ngay ra cache trong `results/`, nên chạy lại không phải trả phí lần nữa.
- **Khoảng tin cậy:** report có CI 95% bootstrap cho từng chỉ số và so sánh cặp (paired bootstrap) với ViT5 + LoRA trên cùng các mẫu; số liệu thô nằm ở `results/main/significance.csv`. Mặc định lấy lại **theo bài gốc, phân tầng theo dataset** (`stats.cluster: article`): 910 mục WikiLingua trong tập test thuộc 318 bài, các mục cùng bài tương quan nên lấy lại theo dòng cho CI hẹp quá mức. Tính lại trên kết quả hiện có: CI ROUGE-L của ViT5 + LoRA rộng thêm ~14% ([26.21, 27.66] thay vì [26.30, 27.57]); mọi so sánh cặp ở tập chính và nhóm 225 văn bản dài giữ nguyên kết luận.
- **Kiểm tra judge bằng người chấm** (không gọi API):

  ```bash
  python scripts/human_eval.py export --config configs/eval.yaml    # -> results/main/human_eval/items.md + annotation_sheet.csv
  # đọc items.md, điền human_score (1–5) vào annotation_sheet.csv
  python scripts/human_eval.py analyze --config configs/eval.yaml   # -> results/main/human_eval/agreement.md
  ```

  Bộ chấm gồm 50 mục lấy ngẫu nhiên (seed) từ các mẫu judge đã chấm, chia đều cho ViT5 + LoRA, ViT5 công khai và LLM zero-shot, ẩn tên hệ thống và điểm judge. `analyze` báo tỷ lệ trùng khớp, tỷ lệ lệch ≤ 1 điểm, Cohen's kappa trọng số bậc hai (kèm CI bootstrap), Spearman và ma trận nhầm lẫn.
