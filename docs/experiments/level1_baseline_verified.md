# Baseline mức 1: đánh giá lại full validation

Ngày 2026-10-24. Cùng evaluator PredCls graph constrained, dùng top-1 predicate
foreground cho mỗi ordered object pair, xếp hạng top K theo xác suất và tính
R@K theo trung bình ảnh, mR@K theo trung bình các predicate có GT support.
Toàn bộ 5.000 ảnh validation được đánh giá độc lập từ `best.pt` của mỗi run;
không lấy 100 ảnh mặc định của script MLP cũ.

Đơn vị **%**, số liệu trung bình ± độ lệch chuẩn mẫu của seeds 42, 123, 3407:

| Mô hình | R@50 | R@100 | mR@50 | mR@100 |
|---|---:|---:|---:|---:|
| Geometry MLP | 49.00 ± 0.08 | 51.59 ± 0.08 | 4.95 ± 0.06 | 5.47 ± 0.08 |
| Semantic MLP | 64.26 ± 0.09 | 66.33 ± 0.02 | 17.67 ± 0.27 | 19.76 ± 0.34 |
| GAT | 64.22 ± 0.14 | 66.35 ± 0.25 | 19.80 ± 1.03 | 22.00 ± 1.07 |

Số chính xác dạng fraction và từng seed ở
`reports/level1/baselines_3seed_val5000_summary.json` và các file
`{mlp,mlp_semantic,gat}_seed{42,123,3407}_val5000.json`.
Đây là report được tạo tại máy hiện tại. Các JSON nhỏ có thể được chia sẻ cùng
tài liệu; checkpoint và dữ liệu gốc vẫn cần được cung cấp riêng để tái tạo.
Teammate có thể tạo lại theo lệnh ở tài liệu quy trình sau khi nhận các
checkpoint tương ứng.

Mọi score chạy lại khớp history tại best epoch. Đây là **validation**, chưa
phải kết quả held-out test. Checkpoint cũ không ghi hash split ở thời điểm
train hoặc max_epochs, nên việc chạy lại trên split hiện tại chưa xác minh được
hoàn toàn provenance dữ liệu/ngân sách lịch sử. Không suy diễn so sánh với
published benchmark khi preprocessing chưa được kiểm chứng tương đương.
