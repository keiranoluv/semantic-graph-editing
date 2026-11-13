# Báo cáo pilot MagicBrush joint-VLM v1

**Mẫu báo cáo — chưa có kết quả inference hay human review.** Không điền tỷ lệ
đúng từ confidence của model hoặc transaction validity.

| Provenance | Giá trị |
|---|---|
| Commit code, run SHA256 | Chưa tạo run |
| Dataset manifest SHA256, revision | Lấy từ run.json |
| Model ID, revision, quantization | Lấy từ config và server.json |
| Runtime versions / source hashes | Lấy từ server.json |
| Prompt/schema version và hash | Lấy từ run.json |
| Seed / batch / pixel/context budget | 42 / 1 / lấy config |
| Pilot IDs / selection strata | Lấy từ run.json; 20 train, không thay ca lỗi |
| Compilation file SHA256 | Chưa compile |
| Human packet / CSV SHA256 | Chưa review |

| Chỉ số | Kết quả |
|---|---|
| Selected | 20 khi prepare |
| Attempted | Chưa chạy |
| Successful complete inference | Chưa chạy |
| JSON-valid / schema-valid | Chưa chạy |
| Graph compilation / 20 | Chưa chạy |
| Transaction valid / 20 | Chưa chạy |
| candidate_accepted | Chưa chạy |
| needs_review / rejected | Chưa chạy |
| Distribution sáu operation, mọi graph compile được | Chưa chạy |
| Các lý do lỗi phổ biến, kể cả retry/interrupted/input lỗi | Chưa chạy |
| Human correct / incorrect / uncertain / unreviewed | Chưa review |
| Đủ 20 determinate và ≥90% correct | Chưa đánh giá |
| Số mẫu train/val dùng được sau lọc | Chưa có bộ expanded |
| Đạt yêu cầu ≥300 mẫu | **Chưa** |

Ghi riêng số liệu structural từ `reports/<compilation-hash>.json`, confidence
VLM từ observations và độ đúng do người thật đánh giá từ `review-audit`.
Trình bày mẫu số rõ ràng; cohort phân tầng theo proxy instruction không đại
diện không thiên lệch cho toàn MagicBrush. Không loại ca khó khỏi báo cáo.

Phân tích lỗi: entity thiếu/sai, correspondence sai, quan hệ bịa, instruction
thất bại, thay đổi ngoài yêu cầu, unsupported style, lỗi JSON/runtime. Thêm
ví dụ đối chiếu hai ảnh và giải thích quyết định người duyệt, không chỉ ghi
transaction pass.

Quyết định tiếp theo: dừng/sửa version mới hoặc mở rộng bằng lệnh riêng sau
cổng thật. Không suy ra candidate=ground truth và không tuyên bố dataset
hoàn thành khi chưa đủ số mẫu cùng các cổng chất lượng.
