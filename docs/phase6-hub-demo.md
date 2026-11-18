# Model trên Hugging Face Hub và demo

← [Về README](../README.md)

```powershell
python scripts/push_to_hub.py --repo_id <tài-khoản>/vit5-vnsum --model_dir C:/vnsum-runs/vit5-vnsum-merged --onnx_dir C:/vnsum-runs/vit5-vnsum-merged-onnx/int8
# chạy thử: chỉ ghi results/main/model_card.md và liệt kê file. Đọc lại, rồi thêm --push (cần huggingface-cli login).
```

Model card sinh từ `results/main/metrics.csv` (`src/vnsum/model_card.py`), gồm số đánh giá, cách dùng, giới hạn và giấy
phép dữ liệu. Repo mới tạo ở chế độ **private**; chỉ thêm `--public` khi đã kiểm tra giấy phép của VietNews. Demo trên
Hugging Face Spaces (Docker, CPU): xem [`demo/README.md`](../demo/README.md).
