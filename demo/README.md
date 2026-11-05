# Demo trên Hugging Face Spaces

Thư mục `space/` là nội dung của một Space kiểu **Docker** (CPU miễn phí): API + UI Streamlit trong một container,
model tải từ Hugging Face Hub khi khởi động.

1. Đưa model lên Hub: `python scripts/push_to_hub.py --repo_id <tài-khoản>/vit5-vnsum --model_dir ... --onnx_dir ... --push`
   (có bản ONNX INT8 thì demo chạy nhanh hơn trên CPU; không có thì dùng PyTorch).
2. Tạo Space mới trên https://huggingface.co/new-space, chọn SDK **Docker**, rồi upload 3 file trong `space/`
   (`README.md`, `Dockerfile`, `start.py`).
3. Settings của Space → Variables: `VNSUM_HF_REPO` = `<tài-khoản>/vit5-vnsum`. Nếu repo model là private, thêm
   Secret `HF_TOKEN` (token chỉ cần quyền đọc).

Lần build đầu mất khoảng 10 phút (cài torch CPU); lần khởi động tải model khoảng 1 GB. Dockerfile clone code từ
GitHub nhánh `main`; đổi `VNSUM_REF` để cố định một commit.

Chưa chạy thử trên Spaces thật: `tests/test_demo.py` chỉ kiểm tra phần tải model và cấu hình API.
