# Mức 1 — Đánh giá VG150 held-out test đã khóa

Chạy từ gốc repository bằng `myenv`. Chủ project chạy GPU; Codex chỉ chuẩn bị
plan và CPU preflight. Đây là **đánh giá checkpoint có sẵn**, không train lại.

## Phạm vi đã khóa trước khi xem test

- 9 điều kiện CLIP × 3 seed = 27 checkpoint, đúng danh sách trong
  [`clip-all-27run-test-plan.json`](../../reports/level1/clip-all-27run-test-plan.json).
- Split test có 26.446 ảnh. Không giới hạn `--max-images`; cả 27 report phải có
  `full_split=true` và cùng hash dữ liệu/cache CLIP.
- Chọn `best.pt` đã có bằng validation mR@50. Plan ghi SHA-256 checkpoint,
  validation report và các file code đọc dữ liệu/đánh giá. Lệnh dừng nếu một
  nguồn bị thay đổi sau lúc khóa.
- Protocol `PredCls_graph_constrained_top1_per_pair`; R@50/100 và mR@50/100,
  mean ± sample standard deviation qua seeds 42, 123, 3407.
- Test dùng để báo cáo khả năng khái quát của **toàn bộ ma trận đã định trước**.
  Nếu sau đó sửa mô hình dựa trên test, các kết quả mới trên cùng test phải ghi
  là exploratory, không gọi là đánh giá held-out mới.

## Lệnh cho Toan chạy

Plan đã được tạo bằng CPU và `--dry-run` PASS. Sau khi commit code, plan và
tài liệu, Toan chạy:

```bash
myenv/bin/python -m tools.level1_test_matrix run --device cuda --max-parallel 3
```

Launcher gọi `scripts.evaluate_level1` với `--split test --allow-test`, batch
size 512 và 0 worker. Có tối đa ba tiến trình evaluate cùng lúc; một tiến trình
xong sẽ khởi chạy ô kế tiếp. Nếu muốn giảm tải GPU/CPU, dùng
`--max-parallel 1` trước khi bắt đầu; không đổi protocol hay checkpoint.

Report từng run được ghi mới trong `reports/level1/test-v1/`, log trong
`logs/level1-clip-test-v1/`. Khi đủ 27 report hợp lệ, launcher tạo
`reports/level1/clip-all-27run-test26446-summary.json` và in bốn metric trung
bình của từng điều kiện. Lệnh từ chối ghi đè report. Nếu mất phiên chạy, gọi
lại cùng lệnh: report hợp lệ được kiểm rồi bỏ qua. Nếu một log tồn tại nhưng
chưa có report, kiểm tra log và lưu nó sang tên khác trước khi chạy lại.

Để xem danh sách lệnh mà không chạy GPU:

```bash
myenv/bin/python -m tools.level1_test_matrix run --device cuda --max-parallel 3 --dry-run
```

Sau khi chạy xong, Codex đối chiếu tất cả report với plan, tính chênh lệch
paired theo seed so với GAT, xem per-predicate/long-tail và đối chiếu
validation. Nếu KG vẫn dưới GAT, phân tích nguyên nhân trên train/validation
và các ví dụ định tính; không dùng test để chọn cấu hình mới rồi báo cùng test
như kết quả cuối.
