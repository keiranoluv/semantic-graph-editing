# Kiểm tra GQA trước đánh giá test

## Kết luận

Sáu lượt mới đã hoàn tất và đạt kiểm tra CPU. Hash checkpoint/source/asset đúng, epoch best khớp cực đại validation mR@50, không có NaN/Inf trong history. Cờ ablation đúng; PE-CE + AlignG không phát sinh PL/PR. Chuỗi lấy mẫu khớp đối chứng trên phần epoch chung. Cả sáu dừng đúng patience 15 của cấu hình cũ.

Các số dưới đây là **validation 5.000 ảnh**, trung bình ± độ lệch chuẩn mẫu trên 3 seed, đơn vị %. Chưa có kết quả test từ launcher mới.

| Model | GC R@50 | GC mR@50 | NG mR@50 |
|---|---:|---:|---:|
| mlp | 76.66 ± 0.59 | 3.33 ± 0.34 | 6.48 ± 0.40 |
| gat | 77.34 ± 0.13 | 3.10 ± 0.10 | 5.81 ± 0.04 |
| pe-ce | 78.07 ± 0.27 | 3.00 ± 0.28 | 6.88 ± 0.50 |
| pe-pl | 78.14 ± 0.07 | 3.42 ± 0.14 | 7.41 ± 0.43 |
| pe-plpr | 77.97 ± 0.07 | 4.16 ± 0.17 | 7.72 ± 0.26 |
| pe-plpr-aligng | 77.95 ± 0.16 | 4.59 ± 0.39 | 9.12 ± 1.11 |
| edge-only | 77.71 ± 0.08 | 4.57 ± 0.25 | 8.55 ± 0.77 |
| pe-ce-aligng | 77.92 ± 0.37 | 3.36 ± 0.10 | 8.14 ± 1.33 |
| gat-kg | 77.32 ± 0.06 | 3.13 ± 0.17 | 5.85 ± 0.50 |

## Hai ablation mới

| Model | Seed | Epoch best | Epoch dừng |
|---|---:|---:|---:|
| edge-only | 42 | 47 | 62 |
| edge-only | 123 | 38 | 53 |
| edge-only | 3407 | 35 | 50 |
| pe-ce-aligng | 42 | 100 | 115 |
| pe-ce-aligng | 123 | 106 | 121 |
| pe-ce-aligng | 3407 | 53 | 68 |

PE-CE + AlignG tăng GC mR@50 khoảng **0,36 pp** so với PE-CE (3,00 → 3,36). Edge-only đạt **4,57%**, sát AlignG đầy đủ **4,59%**; chênh lệch khoảng 0,02 pp chưa đủ để khẳng định GRU prototype update đóng góp rõ rệt. Đây là mô tả thực nghiệm 3 seed, không phải kiểm định ý nghĩa thống kê.

PE-CE + AlignG chọn best ở epoch 100/106 cho hai seed, phù hợp với lo ngại mô hình có thể còn cải thiện muộn. Sáu lượt này vẫn là recipe historical early-stopping; không đổi nhãn thành train đủ 200 epoch. Hai nhánh dự phòng sẽ dùng 200 epoch, không early stopping, cùng đối chứng ngân sách tương ứng.

## Phạm vi test và bằng chứng

- 30 checkpoint nguồn, 27 checkpoint độc nhất; 3 PE-PLPR control có active tensor hash trùng và kết quả CPU smoke trùng. Chỉ lấy best.pt đã chọn trên validation; không lựa chọn epoch bằng test.
- Test project gồm 10.696 ảnh từ official GQA validation; train/validation nội bộ tách từ official train. Không phải challenge test hay reproduction GQA200.
- `training-audit.json`: provenance nguồn, hash, lựa chọn checkpoint, validation và aliases.
- `cpu-smoke.json`: tải/inference 30 checkpoint trên hai ảnh validation, không dùng test.
- `clip-cpu-smoke.json`: CLIP thật trên CPU với ảnh tổng hợp.
- Unit tests: cache/resume/marker, phát hiện shard hỏng, split chồng lấn, dedup hash, mean/std.
- `checkpoint-inventory.csv`: danh sách nguồn và checkpoint canonical.

Lệnh chạy và chính sách resume: [README launcher](../../../tools/gqa_test_v1/README.md). Residual/EMA chỉ là slot dự phòng train/eval, chưa có code model hoặc kết quả; không tự chạy.

Giới hạn kiểm tra: CPU smoke chỉ hai ảnh validation, không thay thế kiểm tra full test hoặc đo VRAM trên GPU. Batch 128 là lựa chọn thận trọng, chưa benchmark trong lượt này. Đã sửa torchvision 0.22.0 → 0.24.1+cu128 để tương thích torch 2.9.1+cu128; torch và code lịch sử giữ nguyên.
