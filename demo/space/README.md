---
title: vnsum — tóm tắt văn bản tiếng Việt
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Demo: tóm tắt văn bản tiếng Việt (ViT5 + LoRA)

Dán một bài báo hoặc văn bản tiếng Việt, nhận 3 bản tóm tắt (ngắn / vừa / dài). Văn bản dài hơn 1,024 token được
tóm tắt theo từng đoạn rồi ghép lại. Chạy trên CPU nên mỗi bản tóm tắt mất vài giây.

Mã nguồn và báo cáo đánh giá: https://github.com/OwenDepTry/vietnamese-long-context-abstractive-text-summarization-system

Model có thể sai sự thật (gán nhầm số liệu, tên, ngày tháng): kiểm tra lại với văn bản gốc trước khi dùng.
