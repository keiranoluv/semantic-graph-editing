# Thực nghiệm dự đoán thao tác MagicBrush

Pipeline đã được chuẩn bị để người dùng chạy một lệnh GPU, tự nối tiếp từ cache CLIP đến sáu lượt huấn luyện, đánh giá test và tổng hợp báo cáo. Kiểm tra dữ liệu thật và kiểm thử mô hình giả lập đã chạy trên CPU; chưa khởi chạy GPU hay tạo kết quả học trên MagicBrush.

Chạy từ terminal của workspace, giữ terminal hoạt động đến khi hoàn tất:

```bash
bash /workspace/workspace/semantic-graph-editing/scripts/run_magicbrush_manual_full602_v1.sh
```

Script mặc định dùng `myenv/bin/python` của repository và W&B offline, không cần đăng nhập mạng để ghi log. Có thể thêm `--wandb-mode online` nếu muốn gửi lên tài khoản W&B đã cấu hình. Lệnh chiếm khóa GPU dùng chung với launcher GQA; nếu một launcher khác đang giữ khóa thì báo lỗi và dừng, không chờ ngầm.

| Giai đoạn tự chạy tiếp | Mô hình hoặc phạm vi | Batch | Seed |
|---|---|---:|---|
| Cache đầu vào train/validation | CLIP ViT-B/32 đóng băng, chỉ câu yêu cầu và graph nguồn | 128 | 42 |
| Huấn luyện MLP encoder + GRU | 301 train, 67 validation | Train 64, validation 32 | 42, 123, 3407 |
| Huấn luyện GNN encoder + GRU | Cùng dữ liệu và decoder | Train 64, validation 32 | 42, 123, 3407 |
| Cache đầu vào test | CLIP ViT-B/32 đóng băng, sau khi chọn xong cả sáu checkpoint | 128 | 42 |
| Đánh giá test và tổng hợp | Từng checkpoint đã chọn; 69 mẫu test | 32 | Theo checkpoint |

**Hàng đợi tự nối tiếp**, chỉ một model chạy tại một thời điểm. Mỗi lượt huấn luyện tối đa 100 epoch, early stopping sau 10 epoch không tăng micro operation F1 trên validation; nếu bằng điểm giữ epoch sớm hơn. AdamW, learning rate 0.0003, weight decay 0.01, clip gradient 5; hidden size 192, byte embedding 64, GNN hai lớp. Decoder dự đoán tối đa 2048 byte-token, không dùng teacher forcing khi validation/test.

Nếu bị ngắt, chạy lại đúng lệnh trên. Lượt đã hoàn thành được kiểm tra checksum rồi bỏ qua; lượt dang dở tiếp tục từ checkpoint epoch đã lưu. Epoch bị ngắt sẽ chạy lại. Pipeline dừng khi có lỗi, không tự đổi batch/config để chạy tiếp dưới cùng mã thực nghiệm.

Output:

- `runs/level2/magicbrush-manual-experiment-v1/summary.json`: kết quả cuối cùng, trung bình và độ lệch chuẩn mẫu qua ba seed.
- `runs/level2/magicbrush-manual-experiment-v1/SUMMARY.txt`: tóm tắt dễ đọc sau khi toàn bộ hàng đợi hoàn tất.
- `runs/level2/magicbrush-manual-experiment-v1/progress.json`: giai đoạn đang chạy, hoàn tất hoặc lỗi.
- `runs/level2/magicbrush-manual-experiment-v1/{mlp,gnn}/{42,123,3407}/`: checkpoint, dự đoán validation/test, kết quả và bằng chứng hoàn tất.
- `data/processed/level2/magicbrush-manual-experiment-v1/clip-{trainval,test}/`: hai cache tách biệt, chứa đầu vào nguồn; không encode graph đích.
- `logs/magicbrush-manual-experiment-v1/launcher.log`: stdout/stderr nối tiếp qua các lần chạy.

Pipeline giữ split theo nhóm nguồn/phiên/ảnh, khóa checksum dữ liệu, code, môi trường và trọng số CLIP. Bằng chứng người dùng xác nhận 20/20 mẫu đúng được đối chiếu với đúng ảnh, event và nội dung graph. Đây là đánh giá đúng/sai tổng thể; không chuyển thành năm đánh giá theo trục chưa được thu thập. Bộ đọc nhãn và bước kiểm tra review mới nằm trong `tools/magicbrush_experiment_v1/`; mã trainer cũ và `src/` được giữ nguyên.

Nhãn test chỉ được đọc bởi bước đánh giá sau khi cả sáu lượt hoàn tất và `selection.json` đã khóa checkpoint theo validation. Kết quả báo precision/recall/F1 cho sáu loại thao tác, exact match tập thao tác và graph đích, exact match nhóm một thao tác, tỷ lệ lỗi decode/transaction, baseline không chỉnh sửa và phân tích theo nhóm thêm/xóa/thay/đổi thuộc tính. Các nhóm có thể chồng lấn; nhóm “đơn giản” được định nghĩa bằng đúng một thao tác đích, không phải đánh giá độ đơn giản ngôn ngữ.

Test có 69/90 cặp được chấp nhận, 21 cặp giữ nháp. Báo cáo giữ rõ mẫu số và danh sách bị loại theo quyết định gán nhãn đã khóa trước thực nghiệm. `replace_edge` có 6 ví dụ train và 0 validation/test: PRF1 tổng hợp của loại không có support được để trống, vẫn giữ số false positive. Các số đo dựa trên nhãn có AI hỗ trợ; kiểm tra tay 20/20 áp dụng cho nhóm train được kiểm tra.

Kiểm tra CPU, không khởi tạo GPU:

```bash
myenv/bin/python -m tools.magicbrush_experiment_v1.pipeline --stage plan
```

Xem [preflight.json](preflight.json), [verification.json](verification.json), [cấu hình](../../../configs/level2/magicbrush-manual-experiment-v1.json) và [báo cáo dữ liệu cùng review](../magicbrush-manual-full602-v1/README.md).
