# Kiểm tay MagicBrush: 20 cặp train cố định

Viewer dùng ảnh nguồn/đích thật và câu yêu cầu của MagicBrush. Hiện **chưa có
graph dự đoán**: hiển thị trạng thái chờ và khóa ô chấm graph, không tự duyệt.
Ảnh nằm dưới `data/`; copy riêng thư mục HTML sẽ không mang theo ảnh.

Mở tại repo root:

```bash
myenv/bin/python -m tools.review_magicbrush --stage serve
```

Truy cập `http://127.0.0.1:8766`, Ctrl+C để dừng. Có thể mở trực tiếp
`index.html` để xem/xuất CSV; nút lưu vào repo chỉ dùng được qua server.

Khi inference đã có, build lại theo
[hướng dẫn pipeline](../../docs/LEVEL2_MAGICBRUSH.md), rồi khởi động lại server.
Mỗi packet bằng chứng được lưu ở `packets/<SHA256>.json`; build không sửa CSV
đã nộp. Đổi graph sẽ đổi packet hash và vùng lưu nháp trình duyệt, không tự
chuyển review cũ sang bằng chứng mới.

Với từng cặp có graph, nhập tên và đánh giá năm trục:

1. Graph nguồn có đúng vật thể/thuộc tính/quan hệ quan trọng trong ảnh nguồn?
2. Graph đích có đúng nội dung ảnh đích?
3. Alignment có giữ đúng instance qua hai ảnh, kể cả nhiều object cùng nhãn?
4. Tập thao tác có diễn tả thay đổi giữa hai graph và tránh thay đổi dư?
5. Thay đổi có phù hợp câu yêu cầu, bao gồm phát hiện edit ngoài schema?

Bbox màu vàng là **dự đoán VLM**, không phải box GT của MagicBrush. Graph và
thao tác tự động vẫn có thể sai dù áp dụng thao tác tái tạo đúng target graph.
Ghi rõ graph bỏ sót, hallucination, nhầm instance hoặc style/toàn cảnh không
biểu diễn được. Đừng coi no-op là đúng khi ảnh có thay đổi chưa được biểu diễn.

Chọn `Đúng`, `Sai`, hoặc `Chưa đủ căn cứ`; không ép trường hợp mơ hồ thành đúng.
Verdict cần tên và đủ năm trục. Chọn tổng thể đúng chỉ khi cả năm trục đúng;
tổng thể sai cần ít nhất một trục sai. Cặp chưa có graph không được chấm.

`Xuất CSV` tải file có pair ID, img_id, turn, split, source/target hash,
evidence/packet hash, đánh giá, reviewer và notes. `Lưu vào reviews/` tạo CSV
bất biến và report ở `submissions/`, không ghi đè các lần lưu trước. Nhập CSV
chỉ chấp nhận cùng packet và ID. Trường trống vẫn trống, không tự approved.

Report đếm đúng/sai/chưa rõ/chưa kiểm và tỷ lệ đúng trong nhóm đã xác định
đúng hoặc sai; khi chưa có kết luận thì tỷ lệ là null. Đây là ước lượng chất
lượng pseudo labels trên mẫu train, không phải metric model hay agreement
giữa nhiều người. Mục tiêu kiểm đủ 20 cặp; nếu graph lỗi, giữ nguyên ID và báo
thiếu bằng chứng, không thay bằng cặp khác để tăng tỷ lệ đúng.

Giữ nguyên mọi phiên bản CSV. Không dùng các quyết định train này để sửa
riêng nhãn validation/test.
