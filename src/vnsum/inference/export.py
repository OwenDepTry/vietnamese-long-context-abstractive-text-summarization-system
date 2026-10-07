"""Export model đã merge sang ONNX (optimum) và lượng tử hóa dynamic INT8 (ORTQuantizer).

Bố cục đầu ra (``onnx_root``):
    fp32/  encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx (+ config, tokenizer)
    int8/  cùng tên file, trọng số INT8 (dynamic quantization, cho CPU)
    fp16/  tùy chọn (--fp16): chuyển từ fp32/ bằng onnxruntime.transformers.float16 (cho GPU)

Không gộp decoder (``no_post_process=True``): decoder_with_past là đường có KV cache, và mỗi file
được lượng tử hóa riêng (tránh subgraph ``If`` của decoder gộp).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EXTRA_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "spiece.model")

# Đoạn văn mẫu tự viết (không trích từ nguồn nào) để kiểm tra nhanh sau export.
SAMPLE_TEXT = (
    "Sáng 6/10, Sở Giao thông vận tải thành phố cho biết tuyến xe buýt điện số 12 sẽ chính thức hoạt động từ đầu tháng 11, "
    "nối bến xe trung tâm với khu công nghệ cao. Tuyến dài 24 km, có 31 điểm dừng, tần suất 10 phút một chuyến vào giờ cao điểm. "
    "Giá vé lượt là 7.000 đồng, học sinh và người cao tuổi được giảm một nửa. Theo đại diện sở, trong ba tháng đầu, "
    "hành khách được đi miễn phí để làm quen với tuyến mới. Đơn vị vận hành đã đưa 20 xe điện vào chạy thử từ tuần trước "
    "và chưa ghi nhận sự cố kỹ thuật. Người dân sống dọc tuyến cho rằng tuyến xe sẽ giúp giảm ùn tắc ở cửa ngõ phía đông."
)

_VI_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def check_vietnamese_output(text: str, min_words: int = 5) -> list[str]:
    """Kiểm tra nhanh output có phải văn bản tiếng Việt hợp lệ; trả danh sách lỗi (rỗng = hợp lệ)."""
    problems = []
    words = text.split()
    if len(words) < min_words:
        problems.append(f"quá ngắn ({len(words)} từ)")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(_VI_DIACRITIC.match(c)) for c in letters) / len(letters) < 0.05:
        problems.append("gần như không có chữ có dấu tiếng Việt")
    if "�" in text or "<unk>" in text or "<pad>" in text:
        problems.append("có ký tự / token lỗi")
    if words and sum(len(w.strip(".,;:!?\"'()-")) <= 1 for w in words) / len(words) > 0.4:
        problems.append("nhiều ký tự / dấu câu rời rạc (không phải câu)")
    if len(words) >= 6:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if len(set(trigrams)) / len(trigrams) < 0.5:
            problems.append("lặp từ nhiều")
    return problems


def _copy_extras(src: Path, dst: Path) -> None:
    for name in _EXTRA_FILES:
        if (src / name).exists() and not (dst / name).exists():
            shutil.copy2(src / name, dst / name)


def _save_tokenizer(model_path: str, out_dir: Path) -> None:
    """Lưu tokenizer (định dạng của transformers đang chạy) vào thư mục ONNX."""
    if not (Path(model_path) / "tokenizer.json").exists():
        logger.warning("%s không có tokenizer.json; thư mục ONNX sẽ không có tokenizer", model_path)
        return
    from vnsum.inference.tokenizer import load_tokenizer_compat

    load_tokenizer_compat(model_path).save_pretrained(str(out_dir))


def ensure_tokenizer(model_path: str, out_dir: Path) -> bool:
    """Thư mục ONNX nạp được tokenizer chưa; nếu chưa thì lưu lại từ model gốc. Trả True nếu đã lưu."""
    from vnsum.inference.tokenizer import load_tokenizer_compat

    if (out_dir / "tokenizer.json").exists():
        try:
            load_tokenizer_compat(out_dir)
            return False
        except Exception as err:
            logger.info("Tokenizer trong %s không nạp được (%s); ghi đè bằng bản lưu lại", out_dir, err)
    _save_tokenizer(model_path, out_dir)
    return True


def export_onnx(model_path: str, out_dir: Path, cfg: dict[str, Any]) -> Path:
    """Export FP32 bằng ``optimum.exporters.onnx.main_export`` (task có KV cache)."""
    from optimum.exporters.onnx import main_export

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "model_name_or_path": model_path,
        "output": out_dir,
        "task": cfg["export"]["task"],
        "no_post_process": True,  # giữ decoder và decoder_with_past riêng
    }
    if cfg["export"].get("opset"):
        kwargs["opset"] = int(cfg["export"]["opset"])
    logger.info("Export %s -> %s (fp32)", model_path, out_dir)
    main_export(**kwargs)
    _save_tokenizer(model_path, out_dir)
    _copy_extras(Path(model_path), out_dir)
    return out_dir


# LayerNorm của T5 (RMSNorm: Pow -> ReduceMean -> Sqrt) giữ ở FP32: bình phương activation dễ vượt
# giới hạn FP16 (65504) -> inf/NaN, đúng lỗi khiến T5 không chạy được ở fp16 trong PyTorch.
FP16_OP_BLOCK_LIST = ["Pow", "ReduceMean", "Sqrt", "Reciprocal"]
# Lần thử 1 (chỉ giữ LayerNorm ở FP32) cho output vô nghĩa trên ViT5. Lần thử 2: giữ thêm ở FP32 các
# phép cộng residual, softmax (T5 không chia điểm attention cho sqrt(d)) và MatMul đầu ra FFN (wo) —
# những chỗ activation của T5 hay vượt 65504. Các MatMul còn lại (phần lớn trọng số) vẫn FP16.
FP16_OP_BLOCK_LIST_STRICT = FP16_OP_BLOCK_LIST + ["Add", "Softmax"]
FP16_NODE_NAME_BLOCK = ("DenseReluDense/wo",)


def convert_fp16(fp32_dir: Path, out_dir: Path, *, strict: bool = True) -> list[Path]:
    """FP32 -> FP16 cho từng file ONNX (input/output giữ FP32), không cần nạp model PyTorch lên GPU.

    Dùng ``onnxruntime.transformers.float16`` (có sẵn trong onnxruntime), thay cho
    ``main_export(device="cuda", dtype="fp16")`` vốn đòi thêm thư viện accelerate.
    """
    import onnx
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    files = sorted(p.name for p in fp32_dir.glob("*.onnx"))
    if not files:
        raise FileNotFoundError(f"Không có file .onnx trong {fp32_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = FP16_OP_BLOCK_LIST_STRICT if strict else FP16_OP_BLOCK_LIST
    block = sorted(set(DEFAULT_OP_BLOCK_LIST) | set(ops))
    written = []
    for name in files:
        model = onnx.load(str(fp32_dir / name))
        nodes = [n.name for n in model.graph.node if strict and any(k in n.name for k in FP16_NODE_NAME_BLOCK)]
        logger.info("FP16 %s (giữ FP32 cho op: %s; %d node theo tên)", name, ", ".join(ops), len(nodes))
        model = convert_float_to_float16(model, keep_io_types=True, op_block_list=block, node_block_list=nodes)
        onnx.save(model, str(out_dir / name))
        written.append(out_dir / name)
    _copy_extras(fp32_dir, out_dir)
    return written


def quantization_config(cfg: dict[str, Any]):
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    q = cfg["export"]["quantization"]
    factory = getattr(AutoQuantizationConfig, q["isa"])
    return factory(is_static=False, per_channel=bool(q.get("per_channel", False)))


def quantize_int8(fp32_dir: Path, out_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    """Dynamic INT8 cho từng file ONNX; giữ nguyên tên file để ORTModelForSeq2SeqLM tự nhận."""
    from optimum.onnxruntime import ORTQuantizer

    files = sorted(p.name for p in fp32_dir.glob("*.onnx"))
    if not files:
        raise FileNotFoundError(f"Không có file .onnx trong {fp32_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    qconfig = quantization_config(cfg)
    written = []
    for name in files:
        logger.info("Quantize %s (%s, dynamic INT8)", name, cfg["export"]["quantization"]["isa"])
        quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name=name)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig, file_suffix="")
        written.append(out_dir / name)
    _copy_extras(fp32_dir, out_dir)
    return written
