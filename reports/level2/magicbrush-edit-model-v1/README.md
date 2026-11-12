# Bước 4: code sẵn sàng, dữ liệu huấn luyện chưa sẵn sàng

- Loader source-graph + instruction, audit split/group/pixel provenance.
- CLIP text đóng băng, cache source/instruction train/val; không encode target.
- MLP và GNN encoder với decoder autoregressive đủ sáu thao tác; thêm no-edit baseline.
- Full-payload evaluator, independent new-ID canonicalization, báo lỗi decode/transaction.
- Khóa hash, human-review gate, checkpoint/resume, W&B model/seed/batch, một run mỗi lệnh.
- **Không chạy GPU, không cache embedding thật, không train dữ liệu thật.**
- Hiện preflight chưa đạt: chưa có full `edits-v1.json` và CSV kiểm tay thật.

`preflight.json` là snapshot CPU hiện tại. `verification.json` ghi kiểm thử CPU,
khả năng bảo toàn thực nghiệm lịch sử và hash mã mới. Các fixture tổng hợp trong
pytest không được dùng làm annotation hoặc kết quả model thật.

Xem [thiết kế, cấu hình và CLI](../../../docs/LEVEL2_EDIT_MODEL.md).
