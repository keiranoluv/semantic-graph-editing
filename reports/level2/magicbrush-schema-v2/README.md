# Sau pilot v1: schema v2 được chuẩn bị, chưa chạy GPU

V1 có 40 raw predictions: 24 PASS cấu trúc, 16 ERROR. CPU compile tạo graph
edit cho 7 cặp, 3 candidate và 4 needs-review; không phải ≥300 nhãn hợp lệ.
Không có review con người hoàn thành ở thời điểm kiểm tra.

V2 giữ model/revision/batch1/seed42/cohort, sửa prompt/schema và cho phép hai
normalization bbox không mất thông tin. Không đoán/xóa cạnh lỗi. Full inference
có gate dựa trên actual human CSV và per-pair evidence, không tự approved.

`verification.json` ghi CPU checks, hash code/cache v1 giữ nguyên, kết quả
retrospective parsing riêng biệt với inference v2. Chưa có prediction hay
human review v2; chưa tuyên bố cải thiện chất lượng model.

Các lệnh theo thứ tự và điều kiện dừng nằm trong
[runbook v2](../../../docs/LEVEL2_MAGICBRUSH_RUN_ORDER_V2.md).
