# GQA project-test v1

Launcher đánh giá **27 checkpoint best trên validation**, thuộc 9 cấu hình × seed
42/123/3407: MLP, GAT, PE-CE, PE-PL, PE-PLPR, PE-PLPR + AlignG, Edge-only,
PE-CE + AlignG, GAT + WordNet v2. Không đánh giá mọi snapshot epoch/last.pt,
không chọn lại checkpoint dựa trên test. Không gồm các pilot hay recipe cũ.

Ba checkpoint PE-PLPR của bộ AlignG có cùng toàn bộ tensor đang hoạt động với
bộ baseline gốc; feedback bị tắt. Chúng được ghi thành alias, không inference lại.
Audit kiểm tra hash nguồn đã khóa, checkpoint, identity, lịch sử hữu hạn, epoch
đạt validation GC mR@50 cao nhất, cờ ablation và CE không dùng PL/PR.

## Chạy

Từ repo root:

```bash
bash tools/gqa_test_v1/run.sh
```

Lệnh tự tạo cache test CLIP một lần rồi **tự chạy tiếp tuần tự cả 27 lượt**, một
model mỗi thời điểm. Batch inference mặc định **128 ảnh**, batch CLIP **64 crop**;
không train và không cần W&B. Không tự chạy Residual/EMA.

- Test: **10.696 ảnh**, official GQA validation được khóa thành project test;
  tách khỏi 69.942 train và 5.000 validation nội bộ. Đây không phải official GQA
  challenge test/GQA200. Từ vựng chỉ lấy từ train: 1.699 object, 310 predicate.
- Cache: `data/cache/gqa-test-clip-v1/`, không ghi đè cache train/validation.
- Output: `reports/level2/gqa-test-v1/`; mỗi model/seed có `result.json`,
  cùng `plan.json` bất biến và `summary.json` (mean ± sample std, đơn vị fraction).
- Metrics: GC/NG R@50/100, mR@50/100 và per-predicate support/recall. Giữ đúng
  evaluator validation: duplicate GT, image-macro, IoU 0.5; mR chia đủ 310 lớp,
  lớp vắng GT tính 0. `normalization-statistics.json` trong cache ghi số OOV bị bỏ.
- Resume: chạy lại cùng lệnh. Shard cache và kết quả hoàn tất có hash đúng sẽ
  được bỏ qua; lượt đang dở chạy lại từ đầu. Không bỏ qua file hỏng/khác provenance.
  Có khóa chống hai launcher ghi cùng cache/output.
- Nếu thiếu VRAM: chạy lại `bash tools/gqa_test_v1/run.sh --batch-size 32`.
  Các lượt hoàn tất vẫn được giữ; batch thực tế được ghi ở từng kết quả.
- Thay code/data/checkpoint đã pin cần phiên bản/output mới, không ghi đè kết quả.

## Kiểm tra CPU

```bash
myenv/bin/python -m unittest tools.gqa_test_v1.test_runner -v
myenv/bin/python -m tools.gqa_test_v1.runner audit
myenv/bin/python -m tools.gqa_test_v1.runner cpu-smoke
```

Smoke dùng hai ảnh **validation**, tải và inference cả 30 checkpoint kể cả alias;
không sử dụng ảnh test. Test cache dùng ảnh tổng hợp và CLIP mock để kiểm tra
resume, identity, split-disjointness và phát hiện corruption. CLIP thật được
kiểm tra riêng trên CPU bằng ảnh đen tổng hợp.

Môi trường hiện tại: torch 2.9.1+cu128, torchvision 0.24.1+cu128. Đã sửa torchvision
0.22.0 không tương thích (lỗi import `torchvision::nms`); không đổi torch hoặc
nguồn đã khóa. Cache pin phiên bản encoder/preprocessing. Nếu tái tạo môi trường:

```bash
myenv/bin/pip install --no-deps torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## Hai nhánh dự phòng

`future_experiments.json` dành chỗ **train/eval** cho AlignG + Residual và EMA:
`enabled=false`, chưa có implementation/checkpoint, không đưa vào hàng đợi hiện tại.
Khi làm mới: namespace/config/output version riêng; **200 epoch, tắt early stopping**,
3 seed, W&B online, train tuần tự. Batch 512 là đề xuất cần kiểm tra capacity.
Cần đối chứng GRU/PE-PLPR cùng ngân sách 200 epoch để so sánh nguyên nhân; không
coi các lượt lịch sử early-stopped là đối chứng ngân sách hoàn toàn tương đương.
Phải định nghĩa rõ EMA state lifecycle trước khi viết model, tránh rò state giữa
ảnh/split. Eval adapter mới dùng lại cache test và thêm output riêng cho checkpoint
mới, không chạy lại 27 lượt này. Ghi nhận thời điểm chốt giả thuyết so với xem test.
