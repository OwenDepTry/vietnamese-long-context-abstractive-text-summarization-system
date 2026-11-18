# Môi trường đã kiểm chứng

← [Về README](../README.md)

| | Train / đánh giá (`.venv`) | Inference ONNX / API (`.venv-onnx`) |
|---|---|---|
| Máy | Windows 11, i7-13620H, RTX 4050 Laptop 6GB | như bên trái |
| Python | 3.13.12 | 3.13 |
| Thư viện chính | torch 2.14.1+cu130, transformers 5.16.1, peft 0.21.2, pandas 2.3.3, rouge-score 0.1.2, underthesea 9.5.0 | transformers 4.57.6, optimum-onnx 0.1.0, onnxruntime 1.23.x (theo `requirements-inference.txt`) |

Cột trái lấy từ `results/long_v2/run_info.json`; từ nay mỗi lần `scripts/evaluate.py` ghi `run_info.json` (commit Git, config, phiên bản thư viện, danh tính model) cạnh kết quả. Hai môi trường tách riêng vì `optimum-onnx` cần transformers < 4.58. CI chạy test offline trên Linux, Python 3.11 và 3.12, không có torch.
