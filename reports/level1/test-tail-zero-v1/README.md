# Phân tích bổ sung VG150 test: long-tail và zero-shot

Long-tail được tính từ 54 report test có sẵn, không inference lại.
Zero-shot chỉ chạy thêm 9 checkpoint fusion đã khóa; mỗi lượt inference 4.660 ảnh có GT zero-shot.
Cohort thuộc toàn bộ test 26.446 ảnh; 7.848 annotation, không lọc prediction trước top-K.

[Bảng long-tail](long-tail.md) · [Bảng zero-shot](zero-shot.md) · [Audit](analysis.json)

## Zero-shot: pair KG thật trừ đối chứng

Delta theo seed 42/123/3407, đơn vị điểm phần trăm. Không phải kiểm định ý nghĩa thống kê.

| Đối chứng | Metric | Mean delta | Seed 42 / 123 / 3407 |
|---|---|---:|---|
| fusion_baseline | zR@50 | -0.288 | +0.692 / -0.192 / -1.366 |
| fusion_baseline | zR@100 | -0.263 | +0.371 / -0.086 / -1.075 |
| fusion_baseline | ng-zR@50 | -0.599 | +0.383 / -0.274 / -1.905 |
| fusion_baseline | ng-zR@100 | +0.169 | +1.145 / +0.301 / -0.941 |
| fusion_pair_shuffled | zR@50 | +0.170 | +0.233 / +0.298 / -0.022 |
| fusion_pair_shuffled | zR@100 | +0.353 | +0.195 / +0.643 / +0.220 |
| fusion_pair_shuffled | ng-zR@50 | +0.152 | +0.096 / +0.147 / +0.211 |
| fusion_pair_shuffled | ng-zR@100 | +0.249 | -0.101 / -0.043 / +0.890 |

## Long-tail: adjusted-off trừ CE-off

| Nhóm | Delta mR@50 (pp) | Seed 42 / 123 / 3407 |
|---|---:|---|
| head | -0.373 | -0.501 / -0.528 / -0.090 |
| body | +10.372 | +10.173 / +11.760 / +9.183 |
| tail | +6.011 | +5.281 / +7.779 / +4.974 |

## Kiểm chứng và phạm vi

- 29 CPU tests pass trước commit/GPU; smoke checkpoint thật so batch 1 và 2.
- Audit độc lập NumPy từ hits/support của cả 9 lượt, khớp 36 metric và mean/sample std.
- Hai ảnh đầu của từng checkpoint được đối chiếu CPU với kết quả GPU đã lưu; cả 9 khớp hits.
- Train vocabulary và head/body/tail giữ nguyên; checkpoint chọn bằng validation legacy, không chọn lại.
- Zero-shot chỉ khảo sát ba cấu hình fusion, không khẳng định đã chạy 54 checkpoint.
- Ảnh không có GT zero-shot không vào mẫu số zR; mỗi ảnh có GT được trọng số bằng nhau.
- Shuffled dùng một permutation; ba seed cùng dấu chưa chứng minh ý nghĩa thống kê.
- Test đã được xem trước đó; không tuyên bố holdout mới. Artifact validation cũ giữ nguyên.
- Không có optimizer, training hoặc W&B trong lượt inference bổ sung này.
