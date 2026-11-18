# Đối chiếu LLM judge với người chấm

Đã chấm 50/50 mục. Mọi số dưới đây tính trên 50 mục đã chấm (một người chấm, rubric giống hệt judge, không biết tên hệ thống và điểm judge khi chấm).

## Mức đồng thuận tổng

| Chỉ số | Giá trị |
|---|---|
| Trùng khớp tuyệt đối | 66.0% |
| Lệch không quá 1 điểm | 84.0% |
| Cohen's kappa trọng số bậc hai | 0.702 (CI 95% bootstrap [0.492, 0.853]) |
| Spearman | 0.733 |
| Điểm TB người / judge | 4.40 / 3.94 |
| Hiệu TB (người − judge) | +0.46 |

## Theo hệ thống

| Hệ thống | n | TB người | TB judge | Trùng khớp | Lệch ≤ 1 |
|---|---:|---:|---:|---:|---:|
| llm_zeroshot | 16 | 5.00 | 5.00 | 100% | 100% |
| vit5_lora | 17 | 5.00 | 4.35 | 65% | 82% |
| vit5_public | 17 | 3.24 | 2.53 | 35% | 71% |

## Ma trận nhầm lẫn (hàng = judge, cột = người)

| judge \ người | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **1** | 1 | 2 | 0 | 0 | 0 |
| **2** | 1 | 2 | 1 | 1 | 2 |
| **3** | 0 | 1 | 1 | 1 | 5 |
| **4** | 0 | 0 | 0 | 1 | 3 |
| **5** | 0 | 0 | 0 | 0 | 28 |

## Các mục lệch từ 2 điểm trở lên (8)

| Mục | Hệ thống | Người | Judge | Lý do của judge | Ghi chú của người |
|---|---|---:|---:|---|---|
| H003 | vit5_public | 5 | 3 | Bản tóm tắt nắm đúng ý chính về cảnh báo hậu quả thảm khốc từ phía Nga liên quan đến phi cơ RC-135. Tuy nhiên, cụm 'hôm qua' không có trong văn bản gốc (gốc ghi ngày 22/5 nhưng không xác định đó là 'hôm qua'), và bản tóm tắt bỏ qua chi tiết quan trọng là nguy cơ va chạm với máy bay dân sự cụ thể. | Hoàn toàn trung thực: phản ánh chính xác nội dung chính của văn bản gốc. |
| H004 | vit5_lora | 5 | 3 | Bản tóm tắt gọi 'thị xã Điện Bàn' thành 'xã Điện Bàn' (sai cấp hành chính) và đơn giản hóa phát ngôn của ông Phong thành 'việc đổi đất là phù hợp' trong khi văn bản gốc chỉ nói việc cân đối diện tích đất tại thời điểm phê duyệt là phù hợp do giá đất thấp lúc đó. | Hoàn toàn trung thực: phản ánh đúng phát biểu của ông Đặng Phong được nêu trong văn bản gốc. |
| H020 | vit5_lora | 5 | 2 | Bản tóm tắt thêm chi tiết 'ở độ cao 10.000 m' không có trong văn bản gốc liên quan đến chuyến bay QF94; độ cao 10.300 m được đề cập trong văn bản là của sự cố máy bay MHS Aviation ở Oman, không phải chuyến bay Qantas QF94. Đây là sự nhầm lẫn mâu thuẫn với văn bản gốc. | Hoàn toàn trung thực: tóm tắt đúng sự cố nhiễu động của máy bay A380 hãng Qantas. |
| H026 | vit5_public | 5 | 3 | Bản tóm tắt nắm đúng ý chính (liệt sĩ Đỗ Phát Hùng mất trí nhớ, trở về quê Thanh Hóa) nhưng bỏ sót các con đi cùng và dùng cụm 'hạnh phúc kỳ lạ' không được văn bản gốc hỗ trợ. Ngoài ra, văn bản gốc không mô tả cảm xúc của chuyến trở về bằng từ 'kỳ lạ'. | Hoàn toàn trung thực: tóm tắt đúng câu chuyện đoàn tụ cảm động của ông Đỗ Phát Hùng. |
| H027 | vit5_public | 4 | 2 | Bản tóm tắt gán cho PGS-TS Trần Đình Thiên chức danh 'Phó Chủ tịch chuyên trách Uỷ ban Kinh tế Trung ương' nhưng văn bản gốc không đề cập chức danh này, đây là thông tin bịa thêm. Phần còn lại về bài học Quảng Ninh và xã hội hóa hạ tầng giao thông có cơ sở trong văn bản gốc. | Gần như trung thực: PGS-TS Trần Đình Thiên là Viện trưởng Viện Kinh tế Việt Nam (hoặc chức danh tương đương) văn bản không nêu cụ thể chức vụ "Phó Chủ tịch chuyên trách Uỷ ban Kinh tế Trung ương". |
| H029 | vit5_public | 5 | 3 | Văn bản gốc đề cập đến tái phát sau một số thủ thuật cụ thể (ví dụ: 50% sau đốt sóng cao tần, vài năm sau dao gamma), nhưng không khái quát rằng bệnh nhân 'thường tái phát nhiều lần sau các thủ thuật phẫu thuật' nói chung. Phần còn lại về cải thiện chất lượng cuộc sống có được văn bản gốc hỗ trợ. | Hoàn toàn trung thực: tóm tắt đúng bản chất và mục đích của các thủ thuật điều trị đau dây thần kinh chia ba. |
| H033 | vit5_public | 5 | 3 | Văn bản gốc chỉ nêu HĐND sẽ 'bầu bổ sung chức danh Phó chủ tịch' mà không xác định cụ thể số lượng là 2 người; dù có đề cập còn thiếu 2 nhân sự, nhưng không khẳng định kỳ họp này sẽ bầu đủ 2 người. | Hoàn toàn trung thực: tóm tắt chính xác nội dung kỳ họp HĐND TP.HCM ngày 21/4. |
| H038 | vit5_lora | 5 | 2 | Ngày 25/8/2017 là ngày ông Guru Mallikarjuna đảm nhiệm chức vụ Tổng giám đốc Bosch Việt Nam thay ông Huệ, không phải ngày ông Huệ 'lên tiếng về việc thay đổi nhân sự'. Thực tế, chính website Bosch Việt Nam đăng thông báo thay đổi nhân sự, còn ông Huệ đăng trên LinkedIn về việc gia nhập Vingroup. | Hoàn toàn trung thực: tóm tắt chính xác thông tin ông Võ Quang Huệ rời Bosch. |

## Lưu ý

- Một người chấm, 50 mục: CI của kappa rộng; đây là kiểm tra độ tin cậy của judge, không phải nghiên cứu inter-annotator agreement đầy đủ.
- Lead-3 không có trong mẫu: trích nguyên văn nên judge luôn chấm 5, không cho thông tin về độ tin cậy.
- Mẫu lấy đều theo hệ thống (không theo tỷ lệ điểm), nên % trùng khớp phụ thuộc phân bố điểm của từng hệ thống.
