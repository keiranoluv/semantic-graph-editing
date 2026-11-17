# Zero-shot VG150 test — 9 checkpoint fusion

GT zero-shot: 7848 annotation trên 4660/26446 ảnh.
Chỉ inference ảnh có GT zero-shot; ranking toàn ảnh giữ nguyên, không lọc prediction.
Mean ± sample std (%), seed 42/123/3407; chưa gặp được định nghĩa từ annotation train.

| Cấu hình | GC zR@50 | GC zR@100 | NG zR@50 | NG zR@100 |
|---|---:|---:|---:|---:|
| fusion_baseline | 11.71 ± 0.91 | 15.25 ± 0.58 | 17.74 ± 1.20 | 28.54 ± 1.40 |
| fusion_pair_real | 11.43 ± 0.17 | 14.98 ± 0.28 | 17.14 ± 0.42 | 28.71 ± 0.36 |
| fusion_pair_shuffled | 11.26 ± 0.01 | 14.63 ± 0.16 | 16.99 ± 0.43 | 28.46 ± 0.35 |
