# Long-tail và zero-shot VG150 test, v1

Không train hoặc chọn lại checkpoint. Không sửa source/manifest của thực nghiệm cũ.

## Phạm vi

- Long-tail: per-predicate recall có sẵn của 54 report test reference-v1 GC,
  18 cấu hình × seed 42/123/3407, 26.446 ảnh. Không inference lại.
- Nhóm head/body/tail 10/20/20 lớp giữ nguyên từ tần suất train đã khóa.
  Kiểm tra trung bình có trọng số theo số lớp khôi phục mR toàn bộ.
- Zero-shot: fusion baseline, pair-real và pair-shuffled × seed 42/123/3407.
  Vocabulary là tất cả bộ ba lớp trong annotation train, gồm GT trùng/đa nhãn.
  Dùng lại vocabulary đã khóa và tái dựng từ train để kiểm tra.
- Chỉ inference ảnh test có GT zero-shot. Ảnh còn lại không đóng góp mẫu số zR;
  vẫn giữ toàn bộ object/candidate/predicate của mỗi ảnh trong ranking.
  Đây không phải lọc prediction theo triplet chưa gặp.
- Đo GC/NG zR@50/100; lưu hits/support từng ảnh để kiểm độc lập image-macro.
  Không suy ra R/mR toàn test từ subset zero-shot.

## Lệnh chạy từ gốc repo

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level1_test_tail_zero_v1 --stage prepare
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level1_test_tail_zero_v1 --stage smoke
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.level1_test_tail_zero_v1 --stage evaluate --device cuda --batch-size 128
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level1_test_tail_zero_v1 --stage summarize
```

`prepare` audit 54 checkpoint/report, tổng hợp long-tail CPU và khóa test cohort/source.
`smoke` kiểm CPU hai ảnh thật với batch 1 và 2.
`evaluate` chạy **tuần tự cả 9 checkpoint**, một model mỗi thời điểm, batch 128,
FP32, TF32 off; không train, không W&B. Tự chuyển lượt kế tiếp rồi dừng khi đủ 9.
Lượt đã có report hợp lệ được bỏ qua; sai hash/split/support bị từ chối.
`summarize` chỉ kiểm và tổng hợp report, không bao giờ khởi chạy inference.

Output: `reports/level1/test-tail-zero-v1/`.

- `long-tail.json`, `long-tail.md`: 18 hàng mean/sample std, support và hash nguồn.
- `test-cohort.json`: chỉ số ảnh, HDF5 row, support GT; không có ảnh/dự đoán model.
- `freeze.json`: 9 checkpoint, report test cũ và hash source.
- `fusion_*-seed*.json`: từng lượt, hits theo ảnh và provenance.
- `zero-shot-summary.json`, `zero-shot.md`: mean/sample std.

Nguồn định nghĩa giữ nguyên ở `long-tail-reference-v1/predicate-groups.json`
và `zero-shot-reference-v1/train-seen-triplets.json`. Nếu source/định nghĩa
thay đổi, tạo phiên bản output mới; không sửa file đã khóa.

Test đã từng được đánh giá, không tuyên bố holdout mới chưa xem. Các phân tích
bổ sung giữ nguyên checkpoint và quy tắc train-only.
