# Kiểm tra sự thật bề mặt (số / ngày / tên riêng)

Sinh bởi `scripts/factcheck_report.py` từ dự đoán đã có. Một bản tóm tắt bị **đánh dấu** khi có số, ngày hoặc tên riêng không tìm thấy trong văn bản gốc (`src/vnsum/factcheck.py`). Chỉ bắt lỗi bề mặt; KHÔNG bắt lỗi quan hệ (gán nhầm lời nói, đảo ý, trộn vai trò).

## Tỷ lệ bị đánh dấu

| Hệ thống | n | Bị đánh dấu (%) |
|---|---:|---:|
| `lead3` | 1,910 | 0.0 |
| `vit5_lora` | 1,910 | 4.4 |
| `vit5_public` | 1,910 | 12.6 |
| `llm_zeroshot` | 1,910 | 4.6 |
| *reference* | 1,910 | 25.4 |

Reference cũng có thể bị đánh dấu: bản tóm tắt gốc của bài báo đôi khi nêu số / tên không có trong thân bài, nên con số này là mức nền chứ không phải 0.

## Đối chiếu với điểm faithfulness của LLM judge (dữ liệu độc lập)

| | n | Faithfulness TB | Judge ≤ 2 (%) |
|---|---:|---:|---:|
| Bị đánh dấu | 51 | 2.73 | 54.9 |
| Không bị đánh dấu | 749 | 4.28 | 14.4 |

Trong các bản judge chấm ≤ 2, bị đánh dấu: **20.6%** (độ phủ). Trong các bản bị đánh dấu, judge chấm ≤ 2: **54.9%** (độ chính xác). Judge có thể sai; đây là đối chiếu, không phải sự thật.

### Quy tắc tên người đầu câu (tùy chọn, mặc định tắt)

| Chế độ | Bị đánh dấu | Độ phủ (%) | Độ chính xác (%) |
|---|---:|---:|---:|
| Bỏ qua từ đầu câu (mặc định) | 51 | 20.6 | 54.9 |
| Kiểm tra tên người đầu câu | 69 | 24.3 | 47.8 |

Chỉ nên bật `check_initial=True` khi độ chính xác của nó không thấp hơn rõ chế độ mặc định.

## Đối chiếu với người chấm (độc lập — không dùng để chỉnh quy tắc)

| | n | Trung thực TB (người) |
|---|---:|---:|
| Bị đánh dấu | 5 | 3.40 |
| Không bị đánh dấu | 45 | 4.51 |

## Ví dụ bị đánh dấu

| Hệ thống | id | Không tìm thấy trong bài |
|---|---|---|
| `vit5_lora` | vietnews-test-19036 | 10-11; Hà Nội Peter Larsen Lucerne Thụy Sĩ |
| `vit5_lora` | vietnews-test-8923 | Chiba Nhật Bản |
| `vit5_lora` | vietnews-test-7664 | Naples Italy |
| `vit5_lora` | vietnews-test-17996 | 80,810 |
| `vit5_lora` | vietnews-test-20715 | Đakrông Quảng Trị |
| `vit5_lora` | vietnews-test-17552 | Ban |
| `vit5_lora` | vietnews-test-3506 | Linh Xuân Hoàng Mai Hà Nội |
| `vit5_lora` | vietnews-test-4414 | 17-11 |
| `vit5_lora` | vietnews-test-8398 | Nga Vladimir Putin |
| `vit5_lora` | vietnews-test-2116 | Nguyễn Thành Phong |
