# MagicBrush schema v2 — cùng 20 cặp train, bằng chứng mới

Chưa có graph inference v2 hoặc review của con người ở thời điểm tạo thư mục.
Không dùng lại quyết định của v1 cho graph khác. ID mẫu vẫn lấy từ cohort đã
khóa trước khi sinh graph, không thay các cặp khó/lỗi bằng mẫu khác.

Các lệnh build/serve sau pilot v2 và thứ tự chạy đầy đủ:
[LEVEL2_MAGICBRUSH_RUN_ORDER_V2.md](../../docs/LEVEL2_MAGICBRUSH_RUN_ORDER_V2.md).

Người duyệt xem ảnh nguồn/đích thật, câu, bbox dự đoán, graph, alignment và
thao tác; điền tên, năm trục, quyết định/notes rồi nhấn **Lưu vào reviews/**.
Graph là pseudo annotation, không phải GT; PASS parser không phải human approval.

`submissions/` giữ CSV và report gốc của từng lần lưu, `packets/` giữ bằng chứng.
Lệnh `tools.magicbrush_review_gate_v2 --select-review` chỉ chọn lần lưu mới
nhất nếu đủ điều kiện. Nó sao chép **nguyên byte** CSV và packet sang
`selected/review.csv`, `selected/packet.json`; không tạo/sửa quyết định, không
chọn một CSV cũ thuận lợi hơn nếu CSV mới chưa đạt. `selected/selection.json`
ghi nguồn/hash. Các bản gốc và bản đã chọn không bị ghi đè khác nội dung.
