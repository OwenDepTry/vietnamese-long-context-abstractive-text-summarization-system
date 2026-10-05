# Phân tích định tính — `vit5_lora`

Chọn từ 194 mẫu đã chấm faithfulness. Document được rút gọn còn 700 ký tự đầu.

## 5 mẫu tốt nhất (faithfulness cao nhất)

### 1. `vietnews-test-17929` (vietnews, short) — faithfulness 5/5, ROUGE-L 85.39

**Document (rút gọn):** Ngày 16-8, tại Đà Nẵng, Tổng cục Du lịch - Bộ Văn hóa - thể thao và du lịch tổ chức hội nghị tăng cường thu hút khách du lịch quốc tế đến Việt Nam. Theo Tổng cục Du lịch, sau một giai đoạn khách du lịch quốc tế đến Việt Nam tăng trưởng đột phá, 7 tháng đầu năm 2019, tăng trưởng khách du lịch quốc tế đã có dấu hiệu chậm lại. Tốc độ tăng trưởng chung bị chậm lại do thị trường Trung Quốc giảm 2,8% so với cùng kỳ 2018. Nguyên nhân do thị trường Trung Quốc giảm theo xu thế chung khi kinh tế nước này khó khăn hơn do ảnh hưởng căng thẳng thương mại Mỹ - Trung. Tổng cục Du lịch cho rằng trước mắt để tăng cường khách du lịch quốc tế đến Việt Nam, cần tập trung khai thác du lịch từ các nhóm thị trường …

**Reference:** Sau một giai đoạn khách du lịch quốc tế đến Việt Nam tăng trưởng đột phá, 7 tháng đầu năm 2019, tăng trưởng khách du lịch quốc tế đã có dấu hiệu chậm lại do thị trường Trung Quốc giảm 2,8% so với cùng kỳ 2018.

**Prediction:** Sau một giai đoạn khách quốc tế đến Việt Nam tăng trưởng đột phá, 7 tháng đầu năm 2019, tăng trưởng khách du lịch quốc tế đã có dấu hiệu chậm lại do thị trường Trung Quốc giảm theo xu thế chung.

**Lỗi cụ thể (unsupported claims theo judge):**
  - (không có)

**Nhận xét của judge:** Mọi thông tin trong bản tóm tắt đều có trong văn bản gốc: tăng trưởng chậm lại trong 7 tháng đầu năm 2019 và nguyên nhân do thị trường Trung Quốc giảm theo xu thế chung đều được nêu rõ.

### 2. `wikilingua_vi-c259afc5e8be-4` (wikilingua_vi, short) — faithfulness 5/5, ROUGE-L 66.67

**Document (rút gọn):** Để mua Bitcoin, ban đầu bạn phải thiết lập kho lưu giữ cho nó, và đây là bước đầu tiên trong quá trình mua Bitcoin. Hiện nay có hai cách để bạn lưu giữ Bitcoin trực tuyến: Giữ chìa khóa của ví trực tuyến. Ví trực tuyến là một tập tin cho phép lưu giữ tiền, tương tự như ví thật ngoài đời. Bạn có thể tạo ví bằng cách cài đặt phần mềm Bitcoin, đó là chương trình để bạn mua bán đồng tiền này. Tuy nhiên, nếu máy vi tính của bạn bị virus hay hacker xâm nhập, hoặc nếu bạn chép nhầm tập tin sang chỗ khác thì Bitcoin có thể bị mất. Luôn luôn sao lưu ví sang một đĩa cứng di động để tránh mất tiền. Lưu giữ Bitcoin thông qua bên thứ ba. Bạn cũng có thể sử dụng ví trực tuyến thông qua một bên thứ ba như  …

**Reference:** Lưu giữ Bitcoin trực tuyến. Tạo ví giấy để giữ Bitcoin. Sử dụng ví cứng để lưu giữ Bitcoin.

**Prediction:** Lưu trữ Bitcoin trực tuyến. Lưu trữ ví giấy Bitcoin. Sử dụng ví cứng.

**Lỗi cụ thể (unsupported claims theo judge):**
  - (không có)

**Nhận xét của judge:** Bản tóm tắt chỉ nêu ba ý chính (lưu trữ trực tuyến, ví giấy, ví cứng) đều có trong văn bản gốc. Không có thông tin nào mâu thuẫn hay không được hỗ trợ.

### 3. `wikilingua_vi-ed7fb5d70283-0` (wikilingua_vi, short) — faithfulness 5/5, ROUGE-L 66.67

**Document (rút gọn):** Khi đã nhiễm virus HSV, bạn cần giữ cơ thể khỏe mạnh nhất bằng cách ăn uống lành mạnh. Tránh ăn các thực phẩm đã qua chế biến, đóng gói hay nấu sẵn. Tăng khối lượng hoa quả, rau, dầu và hạt tiêu thụ mỗi ngày. Hạn chế ăn thịt đỏ, tăng ăn thịt gia cầm không da và cá. Tập trung ăn cacbohydrat phức tạp có trong ngũ cốc nguyên hạt, đậu lăng, đậu và rau. Tránh tiêu thụ các thực phẩm được chế biến sẵn với đường, ví dụ như sirô ngô (rất giàu đường fructose). Nếu bạn cần ăn thứ gì đó ngọt thì nên sử dụng cỏ ngọt, đây là loại thảo mộc ngọt gấp 60 lần đường. Ngoài ra bạn cũng nên tránh chất tạo ngọt nhân tạo. Cơ thể chúng ta hoạt động hiệu quả nhất khi được tập luyện. Bạn nên khởi đầu chậm bằng cách đi …

**Reference:** Ăn uống lành mạnh hơn. Tập thể dục. Thư giãn nhiều hơn.

**Prediction:** Ăn uống lành mạnh. Tập thể dục. Tập yoga.

**Lỗi cụ thể (unsupported claims theo judge):**
  - (không có)

**Nhận xét của judge:** Cả ba điểm trong bản tóm tắt (ăn uống lành mạnh, tập thể dục, tập yoga) đều được đề cập rõ ràng trong văn bản gốc. Không có thông tin nào mâu thuẫn hay không được hỗ trợ.

### 4. `vietnews-test-15479` (vietnews, short) — faithfulness 5/5, ROUGE-L 61.97

**Document (rút gọn):** Video hiện trường phát hiện xác chết người đàn ông. Trưa ngày 5/9, Công an thị xã Thuận An đang phối hợp cùng phòng Cảnh sát Kỹ thuật hình sự, Công an tỉnh Bình Dương điều tra làm rõ nguyên nhân tử vong của một người đàn ông, vừa được phát hiện tại mép sông Sài Gòn, đoạn qua đường Bình Nhâm 83, phường Bình Nhâm, thị xã Thuận An. Trước đó, khoảng hơn 9h cùng ngày, người dân đi qua khu vực trên đã thấy 1 vật lạ đang lơ lửng gần gốc cây, lại gần kiểm tra thì tá hỏa phát hiện 1 người đàn ông đang trong tư thế treo cổ. Sự việc nhanh chóng được tri hô cho người xung quanh và cơ quan chức năng đến ứng cứu. Tuy nhiên, nạn nhân đã tử vong. Công an sau đó đã phong tỏa hiện trường. Bước đầu, xác định n …

**Reference:** Đang đi trên đường, người dân thấy vật lạ đang lơ lửng bên gốc cây, lại gần kiểm tra thì tá hỏa phát hiện xác chết một người đàn ông nên tri hô những người xung quanh.

**Prediction:** Người dân đi qua khu vực trên đã thấy 1 vật lạ đang lơ lửng gần gốc cây, lại gần kiểm tra thì tá hỏa phát hiện 1 người đàn ông đang treo cổ.

**Lỗi cụ thể (unsupported claims theo judge):**
  - (không có)

**Nhận xét của judge:** Toàn bộ thông tin trong bản tóm tắt đều được trích dẫn trực tiếp từ văn bản gốc, không có chi tiết nào bị thêm vào hay mâu thuẫn.

### 5. `vietnews-test-7317` (vietnews, short) — faithfulness 5/5, ROUGE-L 59.09

**Document (rút gọn):** Theo TASS, phòng không Nga vô hiệu hóa đợt tấn công của phiến quân Jabhat al - Nusra vào căn cứ Hmeymim. "Nhóm khủng bố Jabhat al - Nusra đêm qua phóng 6 quả rocket từ vùng đệm giảm căng thẳng ở tỉnh Idlib tới căn cứ không quân Hmeymim. Lực lượng trực chiến phòng không Nga đã bắn hạ toàn bộ các quả đạn này", Bộ Quốc phòng Nga hôm 20/5 cho hay. Nga cho biết không có thiệt hại về người và cơ sở hạ tầng, căn cứ vẫn hoạt động bình thường. Lực lượng phòng không Nga trước đó cũng tiêu diệt hai máy bay không người lái (UAV) của phiến quân trước khi chúng kịp tiếp cận sân bay quân sự Hmeymim. Trước đó, hồi đầu tháng, nhóm Hayat Tahrir al - Sham (HTS) thân al - Qaeda cũng phóng 36 quả rocket vào sân  …

**Reference:** Đợt tấn công của phiến quân Jabhat al - Nusra vào căn cứ Hmeymim đã bị phòng không Nga vô hiệu hóa.

**Prediction:** Lực lượng phòng không Nga vô hiệu hóa đợt tấn công của phiến quân Jabhat al - Nusra vào căn cứ không quân Hmeymim.

**Lỗi cụ thể (unsupported claims theo judge):**
  - (không có)

**Nhận xét của judge:** Bản tóm tắt hoàn toàn phản ánh đúng nội dung văn bản gốc: lực lượng phòng không Nga đã vô hiệu hóa đợt tấn công của phiến quân Jabhat al-Nusra vào căn cứ không quân Hmeymim. Không có thông tin nào mâu thuẫn hay không được hỗ trợ bởi văn bản gốc.

## 5 mẫu tệ nhất (faithfulness thấp nhất)

### 1. `vietnews-test-5654` (vietnews, short) — faithfulness 1/5, ROUGE-L 9.90

**Document (rút gọn):** Khó có ai có thể nghĩ rằng liên minh cầm quyền Barisan Nasional (BN - Mặt trận quốc gia) của Thủ tướng Najib Razak đồng thời cũng là lãnh đạo Đảng UMNO, đã liên tục lãnh đạo Malaysia từ năm 1957, lại có thể bị đánh bại trong lần bầu cử thứ 14 này. Tại sao vậy? Tự đánh mất ưu thế Nhà bình luận chính trị Đông Nam Á Karim Raslan cho rằng chính bản thân BN đã tự đánh mất ưu thế của mình khi tạo ra nhiều nút thắt, khó khăn cho cử tri như việc chọn ngày bầu cử vào ngày trong tuần để làm hạn chế người dân đi bầu cử, đưa luật "chống tin giả" vào áp dụng ngay trước kỳ bầu cử với xu hướng đàn áp hơn là bảo vệ sự thật. Hơn thế nữa, theo nhà báo này, đó chính là sự giận dữ của người dân với tình hình ki …

**Reference:** Chiến thắng của liên minh đối lập Pakatan Harapan (PH - Liên minh Hi vọng) do cựu thủ tướng Mahathir Mohamad dẫn đầu đã khép lại cuộc tổng tuyển cử kịch tính mà tâm điểm là chính trị gia lão luyện 92 tuổi này.

**Prediction:** Ông Mahathir Mohamad, cựu thủ tướng Malaysia, đã tự đánh mất ưu thế của mình khi tạo ra nhiều nút thắt, khó khăn cho cử tri như việc chọn ngày bầu cử vào ngày trong tuần để làm hạn chế người dân đi bầu cử, đưa luật chống tin giả vào áp dụng ngay trước kỳ bầu cử...

**Lỗi cụ thể (unsupported claims theo judge):**
  - Ông Mahathir Mohamad tự đánh mất ưu thế của mình khi tạo ra nhiều nút thắt, khó khăn cho cử tri

**Nhận xét của judge:** Văn bản gốc nêu rõ chính BN (liên minh cầm quyền của Thủ tướng Najib Razak) đã tự đánh mất ưu thế, không phải ông Mahathir. Bản tóm tắt gán hành động này cho ông Mahathir - người thực ra là lãnh đạo phe đối lập PH - là sai lệch nghiêm trọng về chủ thể.

### 2. `vietnews-test-13252` (vietnews, long) — faithfulness 1/5, ROUGE-L 10.10

**Document (rút gọn):** Thế chấp cổ phần hình thành trong tương lai? Theo tìm hiểu của phóng viên báo Đầu tư Bất động sản, dự án The Emerald nằm trên ô đất CT8 có diện tích 19.099 m2 thuộc Khu đô thị mới Mỹ Đình - Mễ Trì, phường Mỹ Đình 1, quận Nam Từ Liêm, Hà Nội. Theo thông tin quảng cáo, Dự án The Emerald có mật độ xây dựng chưa đến 34,7%, diện tích dành cho hệ sinh thái xanh là 36,6%, còn lại là hệ thống đường giao thông nội bộ và hệ thống dịch vụ tiện ích. Theo Quyết định số 1721 / QĐ - UBND ngày 16/5/2009 của UBND TP. Hà Nội, dự án này tiền thân là một dự án tái định cư, thuộc quỹ đất 20% của Khu đô thị mới Mỹ Đình, được giao cho Sở Xây dựng Hà Nội để đầu tư xây dựng nhà tái định cư. Đến ngày 18/1/2010, UBND  …

**Reference:** Quảng bá rầm rộ là "dự án đáng sống nhất tại khu Tây Hà Nội", nhưng dự án The Emerald (quận Nam Từ Liêm, Hà Nội) lại có nhiều nhập nhèm về thế chấp vay vốn trong quá trình triển khai.

**Prediction:** Sau 3 tháng triển khai dự án xây dựng nhà ở tái định cư tại Khu đô thị mới Mỹ Đình - Mễ Trì, Ngân hàng TMCP Việt Á (VietA Bank) tiếp tục tiến hành đăng ký giao dịch đảm bảo cho khoản vay 420 tỷ đồng của Công ty cổ phần Đầu tư bất động sản Mỹ Đình.

**Lỗi cụ thể (unsupported claims theo judge):**
  - Sau 3 tháng triển khai dự án xây dựng nhà ở tái định cư tại Khu đô thị mới Mỹ Đình - Mễ Trì, VietA Bank tiến hành đăng ký giao dịch đảm bảo cho khoản vay 420 tỷ đồng

**Nhận xét của judge:** Văn bản gốc nêu rõ: sau 3 tháng thành lập Công ty cổ phần Đầu tư bất động sản Mỹ Đình (ngày 6/3/2015), VietA Bank đăng ký giao dịch đảm bảo bằng 2 loại tài sản hình thành trong tương lai; còn khoản vay 420 tỷ đồng được đăng ký vào ngày 13/7/2015 - không liên quan đến '3 tháng' hay 'triển khai dự án tái định cư'. Bản tóm tắt gộp hai sự kiện khác nhau thành một, tạo ra thông tin sai lệch nghiêm trọng.

### 3. `vietnews-test-13001` (vietnews, short) — faithfulness 1/5, ROUGE-L 10.81

**Document (rút gọn):** "Cá nhân mình thấy, sau khi giao thừa xong thì mọi người xả rác khá là nhiều. Tại sao xung quanh đấy có rất nhiều thùng rác mà mọi người lại không bỏ vào. Hầu như năm nào mình cũng đi xem nhưng mình có ý thức của bản thân mình là không xả rác. Mình cũng bảo các bạn của mình là không xả rác nhưng có vẻ như hành động của mỗi mình mình không đủ sức mạnh để tác động lên mọi người", đó là ý kiến từ một bạn trẻ về những biển rác này. Thoạt nghe, chúng ta thấy những điều bạn trẻ này nói khá hợp lý. Đó là một mô tuýp quen thuộc khi nói về ý thức của người Việt. "Chúng ta có thùng rác, nhưng chúng ta không bỏ rác vào và vì vậy người Việt có ý thức kém". Nhưng câu hỏi ở đây là, nếu chúng ta có thùng r …

**Reference:** Sáng nay, tôi bật máy tính để đọc báo trong ngày đầu năm mới. Bên cạnh những thông tin về Táo quân, thì là tin về những "bãi rác" khổng lồ ở Hồ Gươm, ở phố đi bộ Nguyễn Huệ sau đêm giao thừa.

**Prediction:** Sau đêm giao thừa, rác được dọn ngay lập tức và tới sáng mùng một thì đường phố lại sạch sẽ, ở bất cứ đâu thuộc về nơi công cộng.

**Lỗi cụ thể (unsupported claims theo judge):**
  - Sau đêm giao thừa, rác được dọn ngay lập tức và tới sáng mùng một thì đường phố lại sạch sẽ, ở bất cứ đâu thuộc về nơi công cộng

**Nhận xét của judge:** Văn bản gốc nêu đây chỉ là một giả thiết ('tôi giả sử thế') và ngay sau đó tác giả khẳng định ngược lại: 'Hà Nội chưa bao giờ sạch sẽ, ở bất cứ đâu thuộc về nơi công cộng'. Bản tóm tắt trình bày giả thiết như một sự thật, mâu thuẫn trực tiếp với nội dung chính của văn bản.

### 4. `wikilingua_vi-eed55baa22f6-3` (wikilingua_vi, short) — faithfulness 1/5, ROUGE-L 14.49

**Document (rút gọn):** Sẽ có người nhận ra hai người không còn là bạn nữa và hỏi chuyện gì đã xảy ra. Bạn có thể đáp lại với những câu trả lời chung chung, kiểu như “chúng tôi không còn dành được nhiều thời gian bên nhau nữa”, chứ đừng đưa ra những câu trả lời chi tiết. Nói xấu người đã từng là bạn mình là một việc vô cùng tồi tệ và ấu trĩ, bất kể lý do gì khiến tình bạn giữa hai người tan vỡ. Nếu người bạn của bạn trở nên khó chịu, lan truyền tin đồn hoặc buôn chuyện về bạn trên mạng xã hội, hãy cố không tham gia vào những câu chuyện đó. Việc xen vào câu chuyện hay tự bảo vệ bản thân khỏi một người mà bạn không còn muốn làm bạn cùng nữa là một việc hoàn toàn vô nghĩa. Nhưng quan trọng nhất là bạn không làm điều t …

**Reference:** Đừng “bà tám” nếu có ai hỏi bạn về những gì đã xảy ra. Lịch sự mỗi khi hai người gặp mặt. Đừng bị cuốn vào chuyện rắc rối nào nếu bạn bè chung của hai người trở nên bất hòa.

**Prediction:** Nói xấu người đã từng là bạn của bạn. Tôn trọng những gì bạn đã từng có. Hãy tôn trọng cô ấy. Không kết thúc một tình bạn.

**Lỗi cụ thể (unsupported claims theo judge):**
  - Nói xấu người đã từng là bạn của bạn (văn bản gốc khuyên KHÔNG nên nói xấu, bản tóm tắt lại đề xuất làm điều đó)
  - Không kết thúc một tình bạn (văn bản gốc hướng dẫn cách kết thúc tình bạn, không phải khuyên không nên kết thúc)

**Nhận xét của judge:** Bản tóm tắt mâu thuẫn trực tiếp với văn bản gốc: văn bản gốc khuyên không nên nói xấu bạn cũ, nhưng tóm tắt lại đề xuất 'nói xấu người đã từng là bạn'; đồng thời văn bản gốc hướng dẫn cách kết thúc tình bạn, nhưng tóm tắt lại nói 'không kết thúc một tình bạn'.

### 5. `vietnews-test-18644` (vietnews, short) — faithfulness 1/5, ROUGE-L 14.55

**Document (rút gọn):** Với vốn điều lệ trên 9.800 tỉ đồng và giá tham chiếu ngày giao dịch đầu tiên là 33.000 đồng / cổ phiếu, HDBank có vốn hóa 32.373 tỉ đồng (1,43 tỉ USD) và nằm trong nhóm 20 cổ phiếu vốn hóa lớn nhất HOSE. Trưa nay, giá cổ phiếu HDB ở mức 38.900 đồng / cổ phiếu, tăng gần 18% so với giá tham chiếu, khối lượng giao dịch đạt hơn 19,22 triệu cổ phiếu. Với việc niêm yết ngay những ngày đầu năm 2018, HDBank cũng mở màn cho các doanh nghiệp vốn hóa lớn sẽ lên sàn chứng khoán Việt Nam trong năm nay. Mới đây HOSE thông báo đã nhận được hồ sơ niêm yết cổ phiếu của Ngân hàng Tiên Phong (TPBank), mã chứng khoán là TPB. Thời gian TPBank dự kiến niêm yết là cuối quý 1-2018, sau khi hoàn tất đợt phát hành tă …

**Reference:** Hơn 980 triệu cổ phiếu HDBank (mã HDB) đã chính thức giao dịch trên sàn chứng khoán TP. HCM (HOSE) hôm nay, 5-1, với giá tham chiếu 33.000 đồng / cổ phiếu.

**Prediction:** Ngày 5-1, Ngân hàng TMCP Tiên Phong (TPBank) thông báo đã nhận được hồ sơ niêm yết cổ phiếu của HDBank.

**Lỗi cụ thể (unsupported claims theo judge):**
  - TPBank thông báo đã nhận được hồ sơ niêm yết cổ phiếu của HDBank

**Nhận xét của judge:** Bản tóm tắt hoàn toàn sai lệch: theo văn bản gốc, chính HOSE (Sở Giao dịch Chứng khoán TP.HCM) nhận được hồ sơ niêm yết của TPBank, không phải TPBank nhận hồ sơ của HDBank. Chủ thể và đối tượng đều bị gán sai.
