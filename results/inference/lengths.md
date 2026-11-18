# So sánh mức độ dài short / medium / long

Sinh tự động lúc 2026-10-23 10:24 UTC bởi `scripts/compare_lengths.py`. 50 tài liệu test cố định (seed 42), backend pytorch bf16 cuda, chiến lược `extractive`; latency là sau 1 lần warmup.

| Mức | Tham số | Token sinh TB (trung vị) | Âm tiết TB | Latency TB (ms) | ROUGE-L | Lặp 3-gram (%) |
|---|---|---|---|---|---|---|
| `short` | max_new_tokens=48, length_penalty=0.6 | 28.3 (29) | 23.1 | 2,127 | 26.05 | 0.00 |
| `medium` | max_new_tokens=128 | 30.6 (31) | 24.9 | 2,234 | 27.27 | 0.00 |
| `long` | max_new_tokens=256, min_new_tokens=64, length_penalty=2.0 | 70.9 (69) | 60.3 | 4,677 | 23.03 | 44.45 |

## Hai mức cho cùng kết quả

| Cặp | Số tài liệu giống hệt |
|---|---|
| `short` = `medium` | 33/50 |
| `short` = `long` | 0/50 |
| `medium` = `long` | 0/50 |

## Ví dụ (3 tài liệu đầu)

### wikilingua_vi-81bfec70d7f4-0

**Reference:** Đong nguyên liệu cho vào máy xay. Điều chỉnh màu sắc của máu giả. Làm đặc máu giả.

- **short** (11 token): Chuẩn bị nguyên liệu. Chuẩn bị hỗn hợp.
- **medium** (26 token): Chuẩn bị nguyên liệu. Thêm một ít nước hoa quả màu đỏ vào máy xay sinh tố. Thêm xi-rô ngô.
- **long** (68 token): Chuẩn bị nguyên liệu. Thêm một ít nước hoa quả màu đỏ vào máy xay sinh tố. Thêm xi-rô ngô vào hỗn hợp. Thêm vài giọt màu thực phẩm đỏ. Thêm nước hoa hồng hoặc bột ca cao vào máu giả. Cho thêm một ít bột ngô vào nước. Thêm thêm một chút bột ngô và trộn đều.

### vietnews-test-21762

**Reference:** Dây điện nối micro với dàn karaoke bị trục trặc, ông Đủ đến chỉnh sửa thì bị điện giật tử vong.

- **short** (25 token): Trong lúc hát, dây điện nối micro với dàn karaoke bị trục trặc nên ông Đủ bị điện giật văng ra ngoài.
- **medium** (26 token): Trong lúc hát karaoke, dây điện nối micro với dàn karaoke bị trục trặc nên ông Đủ bị điện giật văng ra ngoài.
- **long** (83 token): Trong lúc hát karaoke, dây điện nối micro với dàn karaoke bị trục trặc nên ông Đủ bị điện giật văng ra ngoài, người hát chung chạy đến đỡ ông lên nhưng cũng bị văng xuống sàn nhà, thoát chết trong gang tấc, may mắn thoát chết kịp thời, không ai kịp cứu nạn nhân thoát chết. Cảnh sát đã đến lập biên bản, bàn giao thi thể cho gia đình lo hậu sự.

### wikilingua_vi-7b1913a6e30e-3

**Reference:** Học cách để tập trung vào sở thích của người khác. Gặp gỡ người mà bạn cảm thấy rằng họ khá thú vị. Mỉm cười và cười vang càng thường xuyên càng tốt. Học cách để phớt lờ thái độ xúc phạm hoặc thiếu tôn trọng của người khác.

- **short** (22 token): Hãy để ý đến người mà bạn đang trò chuyện. Hãy mỉm cười và cười vang. Cười vang.
- **medium** (22 token): Hãy để ý đến người mà bạn đang trò chuyện. Hãy mỉm cười và cười vang. Cười vang.
- **long** (68 token): Hãy để ý đến người mà bạn đang trò chuyện. Hãy mỉm cười và cười vang. Chấp nhận những người có sở thích và sở thích khác nhau. Cười vang. Đừng cố gắng phóng đại lời lăng mạ. Đừng bao giờ nói "Tôi đang học trượt tuyết nhanh hơn bạn". Không bao giờ tỏ ra quá thô lỗ.
