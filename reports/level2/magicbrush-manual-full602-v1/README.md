# Dữ liệu MagicBrush và kết quả kiểm tra tay

Đã xử lý đủ 602 cặp ảnh MagicBrush trong manifest của dự án, tương ứng 912 tệp ảnh. Bộ nhãn được chấp nhận gồm 437 cặp; 165 cặp còn mơ hồ hoặc lỗi ảnh được giữ riêng dưới dạng nháp, kèm lý do.

| Phân chia | Được chấp nhận | Giữ nháp |
|---|---:|---:|
| Train | 301 | 114 |
| Validation | 67 | 30 |
| Test | 69 | 21 |
| Tổng | 437 | 165 |

Người thực hiện kiểm tra tay 20 mẫu được chọn trước trong tập train và xác nhận 20/20 mẫu đúng, 0 mẫu sai, 0 mẫu chưa chắc. Tỷ lệ đúng quan sát được trên nhóm kiểm tra là **100% (20/20)**. Kết quả do người dùng xác nhận trong hội thoại, được gắn với đúng ID, checksum ảnh và phiên bản nhãn trong [bằng chứng kiểm tra tay](human-review.json). Đây là đánh giá đúng/sai tổng thể trên nhóm 20 mẫu; chưa đo riêng từng trục chất lượng hoặc độ đồng thuận giữa nhiều người gán nhãn.

Nhãn được tạo với hỗ trợ của ChatGPT/Codex và kiểm tra trên cả ảnh nguồn, câu yêu cầu, ảnh đích. Graph mô tả các đối tượng và quan hệ nổi bật; một số nhóm ngữ cảnh không đổi được biểu diễn bằng một node. Nhãn phản ánh thay đổi nhìn thấy, có thể gồm thay đổi ngoài yêu cầu hoặc yêu cầu chỉ được đáp ứng một phần. Ghi chú từng mẫu lưu các trường hợp này.

Toàn bộ 602 cặp graph qua kiểm tra cấu trúc và áp dụng edit tái tạo đúng graph đích. Cả 437 cặp được chấp nhận đạt giới hạn codec; checksum ảnh và phân chia theo nhóm nguồn, phiên chỉnh sửa, pixel đã được kiểm tra. Loader đọc đủ 301 mẫu train và 67 mẫu validation. Nhãn test được xuất riêng; đánh giá phải công bố coverage 69/90 mẫu test, cùng 21 mẫu bị loại.

Đường dẫn bản xuất, checksum và số lượng từng thao tác nằm trong [summary.json](summary.json). Xác nhận kiểm tra tay này bổ sung cho các báo cáo trước đó còn ghi chờ human review; các bản xuất và audit lịch sử được giữ nguyên.

Bước tiếp theo là chuẩn bị thực nghiệm dự đoán thao tác từ **graph nguồn và câu yêu cầu**:

1. Tạo cấu hình phiên bản mới trỏ đến bản xuất 368 mẫu train/validation và bằng chứng kiểm tra tay. Nối kết quả đúng/sai thực tế với bước kiểm tra đầu vào của launcher; không suy diễn điểm theo từng trục chưa được thu thập.
2. Kiểm tra trên CPU việc đọc dữ liệu, codec, đầu vào chỉ từ nguồn và đánh giá. Mã hiện có hỗ trợ CLIP text đóng băng, MLP encoder + GRU, GNN encoder + GRU và baseline không chỉnh sửa.
3. Hoàn tất code và commit trước khi bàn giao lệnh GPU. Dự kiến chạy hai mô hình với seed 42, 123, 3407; người dùng chạy tuần tự một model mỗi thời điểm. Batch size, output và cơ chế hàng đợi sẽ được chốt trong launcher mới.
4. Chọn checkpoint bằng validation rồi đánh giá test; báo cáo precision/recall/F1 theo sáu loại thao tác, khớp hoàn toàn tập thao tác cho yêu cầu đơn giản, tỷ lệ output lỗi và trung bình/độ lệch chuẩn qua ba seed.

Thao tác `replace_edge` hiện có 6 ví dụ train và không có ví dụ validation/test. Báo cáo phải nêu support bằng 0 và đánh dấu các chỉ số không xác định phù hợp; tập này chưa đo được khả năng tổng quát hóa riêng cho thao tác đó. Các mẫu đơn giản cần có tiêu chí công bố trước; quy ước hiện tại “một thao tác đích” chỉ là đại diện ở mức graph.

Phần MagicBrush đã đủ số lượng cặp tối thiểu và có kết quả kiểm tra tay được ghi nhận theo đề bài. Huấn luyện và đánh giá trên bản dữ liệu này là công việc tiếp theo; chưa có kết quả mô hình từ bản phát hành này.

Pipeline thực hiện các bước tiếp theo đã được chuẩn bị riêng tại [hướng dẫn thực nghiệm một lệnh](../magicbrush-manual-experiment-v1/README.md), gồm kiểm tra CPU, hàng đợi MLP/GNN ba seed, tiếp tục khi bị ngắt và đánh giá test sau khi khóa checkpoint.
