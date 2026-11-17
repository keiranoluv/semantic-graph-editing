# VG150 reference-v1: test của 27 checkpoint đã khóa

Chạy từ gốc repository:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level1_reference_test27_v1 --dry-run
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.level1_reference_test27_v1 --batch-size 128
```

Phạm vi: 9 cấu hình MLP/GAT/KG ban đầu × seeds 42/123/3407, đủ 26.446 ảnh test,
reference-v1 GC. Batch 128, một checkpoint mỗi thời điểm; tự chạy tiếp hàng đợi
đến đủ 27 rồi dừng. Không train hoặc chọn lại checkpoint. Test đã từng được xem;
đây là tái chấm bằng evaluator mới. Không chạy lại legacy test, không gồm pair/loss/fusion.

Output: `reports/level1/reference-test27-v1/`, mỗi run một JSON và `summary.json`
mean/sample std ba seed. Launcher kiểm SHA256 27 checkpoint với plan lịch sử trước
inference. Chạy lại cùng lệnh sẽ kiểm và bỏ qua report hợp lệ, gồm hash evaluator,
split, số ảnh, dữ liệu, cache, checkpoint và seed. Report không tương thích hoặc
JSON dở dang khiến launcher dừng để kiểm tra, không ghi đè hay âm thầm chạy lại.
Lock ngăn hai launcher phiên bản này cùng chạy. Không khởi động launcher khác đồng thời.

Đây là evaluation, không yêu cầu W&B. Provenance của evaluator được lưu trong từng
report; summary liên kết SHA256 report và plan. Giữ nguyên source/config/manifest
và kết quả legacy. Dry-run chỉ kiểm CPU và in các lệnh còn thiếu.
