# Kế hoạch bàn giao sau khi hoàn thiện data card VN100

Ghi chú lập kế hoạch. VN100 hiện đã phát hành trên Hugging Face. Hướng dẫn dùng bản nộp hiện hành nằm trong [README](../../README.md) và [hướng dẫn tái lập](../../docs/REPRODUCE.md).

1. Viết lại README gốc với các lối vào cho demo, dữ liệu và mã nguồn. Nêu rõ phần nào chạy checkpoint, phần nào phát lại dự đoán đã lưu.
2. Tập hợp hướng dẫn tái lập theo nhiệm vụ. Mỗi mục có môi trường, lệnh, checkpoint, split hoặc mã tạo split, đơn vị chia tập, kiểm tra rò rỉ và kết quả kỳ vọng. Tách môi trường CPU của demo khỏi các launcher GPU.
3. Lập danh sách tệp đóng gói và artifact ngoài Git, kèm đường dẫn khôi phục, dung lượng và SHA-256. Giữ đường dẫn của thí nghiệm đã khóa. Kiểm tra import và tham chiếu trước khi loại pilot, audit hoặc note khỏi bản nộp.
4. Kiểm thử từ bản clone sạch: mở được output đã lưu, chạy được ca MagicBrush xóa ghế khớp cả thao tác lẫn đồ thị đích, sau đó chạy đủ hai bảng 5 Hit/5 Miss và sáu ca VN100. Ghi commit và checksum của bản đã kiểm thử.
5. Khi có bản nộp đã xác minh, quyết định dùng gói nộp hay repo xuất bản riêng. Giữ lịch sử nghiên cứu và các tệp cá nhân ở workspace.

Tài liệu đang xử lý: [Data card VN100](../../docs/LEVEL3_DATA_CARD.md).
