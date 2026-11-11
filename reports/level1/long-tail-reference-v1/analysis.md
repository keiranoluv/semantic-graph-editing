# Long-tail Mức 1 — cùng checkpoint, cùng ranking, chia nhóm theo train

54 checkpoint / 18 cấu hình × 3 seed (42, 123, 3407), cùng 5.000 ảnh validation. GC và NG được tổng hợp riêng từ report đã có; không train, không inference lại, không chọn lại checkpoint.

## Cách chia và cách tính

Đếm tất cả annotation quan hệ trong 57.723 ảnh train, giữ dòng trùng và cặp nhiều predicate. Xếp tần suất giảm dần; đồng hạng xếp predicate ID tăng dần. Head = hạng 1–10, body = 11–30, tail = 31–50. Đây là quy ước phân tích của dự án, không khẳng định mọi paper dùng cùng nhóm.

mR nhóm = trung bình recall theo ảnh của từng predicate trong nhóm. Lớp vắng GT validation vẫn đóng góp 0 và nằm trong mẫu số cố định. Top-K giữ nguyên danh sách dự đoán của **toàn ảnh**; không lọc nhóm rồi lấy lại top-K. Không tính được R nhóm theo ảnh chỉ từ các report tổng hợp này, nên chỉ báo **mR theo nhóm**.

Với từng checkpoint/protocol/K, đã kiểm `0,2 × head-mR + 0,4 × body-mR + 0,4 × tail-mR = mR toàn bộ 50 lớp` (sai số ≤ 1e-12). Không lấy trung bình không trọng số của ba nhóm có kích thước khác nhau.

| Nhóm | Số lớp | Annotation train | Tỷ trọng train | Annotation val | Lớp có GT val |
|---|---:|---:|---:|---:|---:|
| head | 10 | 358780 | 88.40% | 29468 | 10 |
| body | 20 | 41145 | 10.14% | 3238 | 20 |
| tail | 20 | 5935 | 1.46% | 497 | 19 |

Lớp vắng GT validation, vẫn tính 0 trong nhóm: flying in.

## Kết quả — đơn vị %, mean ± sample std

### main / GC

| Cấu hình | Head mR@50 | Body mR@50 | Tail mR@50 | Head mR@100 | Body mR@100 | Tail mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| mlp_semantic | 46.31 ± 1.77 | 17.41 ± 1.87 | 2.45 ± 0.35 | 50.48 ± 1.31 | 20.33 ± 1.96 | 3.25 ± 0.67 |
| gat | 47.96 ± 0.64 | 21.03 ± 1.16 | 3.87 ± 0.52 | 51.70 ± 0.95 | 23.24 ± 1.19 | 4.41 ± 0.84 |
| kg_off | 47.77 ± 0.65 | 20.32 ± 0.15 | 3.31 ± 0.85 | 51.37 ± 0.60 | 22.68 ± 0.50 | 4.03 ± 0.37 |
| wordnet_kg_on | 47.16 ± 0.67 | 19.30 ± 0.97 | 3.27 ± 0.64 | 51.13 ± 0.24 | 21.83 ± 0.76 | 3.83 ± 0.83 |
| wordnet_edges_off | 48.64 ± 0.85 | 21.30 ± 2.46 | 3.06 ± 1.08 | 52.15 ± 1.02 | 23.49 ± 2.00 | 3.79 ± 1.39 |
| conceptnet_transe_kg_on | 47.91 ± 0.54 | 20.33 ± 1.00 | 3.99 ± 0.83 | 51.55 ± 0.34 | 22.65 ± 1.29 | 4.89 ± 0.97 |
| conceptnet_transe_edges_off | 48.54 ± 0.68 | 20.96 ± 1.49 | 3.49 ± 1.00 | 52.16 ± 0.68 | 23.29 ± 1.35 | 3.93 ± 1.00 |
| conceptnet_rotate_kg_on | 47.61 ± 0.59 | 19.87 ± 1.47 | 3.21 ± 0.17 | 51.21 ± 0.96 | 21.81 ± 1.09 | 3.62 ± 0.54 |
| conceptnet_complex_kg_on | 47.78 ± 0.64 | 19.54 ± 1.09 | 4.19 ± 0.61 | 51.44 ± 0.66 | 22.11 ± 1.24 | 4.72 ± 0.42 |

### main / NG

| Cấu hình | Head ng-mR@50 | Body ng-mR@50 | Tail ng-mR@50 | Head ng-mR@100 | Body ng-mR@100 | Tail ng-mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| mlp_semantic | 58.86 ± 3.16 | 34.87 ± 2.64 | 14.12 ± 2.15 | 71.73 ± 2.67 | 48.24 ± 3.35 | 22.79 ± 3.47 |
| gat | 61.56 ± 0.67 | 37.95 ± 0.77 | 14.30 ± 0.97 | 74.17 ± 0.85 | 51.00 ± 1.42 | 21.57 ± 3.56 |
| kg_off | 61.57 ± 0.24 | 36.73 ± 0.35 | 12.92 ± 3.50 | 74.19 ± 0.45 | 50.78 ± 0.97 | 22.09 ± 2.03 |
| wordnet_kg_on | 60.69 ± 1.59 | 37.08 ± 0.79 | 12.17 ± 2.62 | 73.69 ± 1.27 | 50.96 ± 1.62 | 21.86 ± 1.00 |
| wordnet_edges_off | 62.17 ± 1.10 | 37.75 ± 0.59 | 12.33 ± 3.51 | 74.87 ± 1.26 | 51.73 ± 0.80 | 20.43 ± 4.05 |
| conceptnet_transe_kg_on | 61.40 ± 0.73 | 37.98 ± 1.90 | 10.08 ± 0.73 | 74.29 ± 0.49 | 51.24 ± 2.24 | 23.10 ± 0.37 |
| conceptnet_transe_edges_off | 62.56 ± 0.62 | 37.03 ± 1.34 | 13.14 ± 3.15 | 75.01 ± 0.46 | 52.47 ± 0.63 | 23.62 ± 0.90 |
| conceptnet_rotate_kg_on | 61.36 ± 0.20 | 37.43 ± 0.83 | 14.45 ± 1.18 | 73.87 ± 0.37 | 51.84 ± 1.07 | 21.99 ± 1.21 |
| conceptnet_complex_kg_on | 61.29 ± 0.67 | 37.71 ± 0.77 | 14.19 ± 3.89 | 73.99 ± 0.73 | 52.20 ± 0.55 | 23.34 ± 1.11 |

### pair / GC

| Cấu hình | Head mR@50 | Body mR@50 | Tail mR@50 | Head mR@100 | Body mR@100 | Tail mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| ce_off | 47.64 ± 0.35 | 20.47 ± 0.99 | 3.65 ± 0.60 | 51.54 ± 0.32 | 22.30 ± 0.72 | 4.18 ± 0.56 |
| ce_on | 47.94 ± 1.07 | 20.52 ± 2.69 | 4.88 ± 0.48 | 51.66 ± 1.00 | 22.95 ± 3.10 | 5.41 ± 0.70 |
| adjusted_off | 46.85 ± 0.13 | 31.28 ± 1.58 | 9.79 ± 2.33 | 50.73 ± 0.24 | 33.84 ± 1.53 | 10.50 ± 2.36 |
| adjusted_on | 46.34 ± 0.70 | 33.24 ± 1.27 | 8.51 ± 0.30 | 50.13 ± 0.49 | 36.41 ± 1.48 | 9.73 ± 0.50 |

### pair / NG

| Cấu hình | Head ng-mR@50 | Body ng-mR@50 | Tail ng-mR@50 | Head ng-mR@100 | Body ng-mR@100 | Tail ng-mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| ce_off | 60.89 ± 0.29 | 37.55 ± 1.43 | 12.64 ± 2.50 | 73.47 ± 0.36 | 50.70 ± 0.45 | 23.70 ± 0.71 |
| ce_on | 61.99 ± 0.67 | 37.71 ± 0.86 | 12.18 ± 4.25 | 74.09 ± 0.26 | 51.49 ± 1.26 | 22.88 ± 0.65 |
| adjusted_off | 59.00 ± 0.66 | 43.46 ± 1.61 | 17.19 ± 3.51 | 71.82 ± 0.29 | 55.11 ± 1.41 | 28.81 ± 1.02 |
| adjusted_on | 58.78 ± 0.82 | 44.34 ± 0.63 | 18.67 ± 3.08 | 71.25 ± 0.73 | 56.76 ± 1.93 | 28.30 ± 0.91 |

### fusion / GC

| Cấu hình | Head mR@50 | Body mR@50 | Tail mR@50 | Head mR@100 | Body mR@100 | Tail mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 46.22 ± 1.17 | 32.39 ± 2.60 | 8.95 ± 0.61 | 50.25 ± 0.59 | 35.41 ± 2.38 | 10.52 ± 0.58 |
| pair_real | 47.25 ± 0.23 | 32.20 ± 0.45 | 8.46 ± 0.65 | 50.97 ± 0.50 | 34.92 ± 0.74 | 9.40 ± 0.37 |
| pair_shuffled | 46.64 ± 0.07 | 31.60 ± 1.71 | 9.70 ± 1.84 | 50.41 ± 0.13 | 34.55 ± 1.36 | 10.98 ± 2.23 |
| visual_real | 45.61 ± 0.20 | 32.32 ± 0.52 | 7.87 ± 0.92 | 49.57 ± 0.05 | 35.21 ± 0.92 | 9.21 ± 0.88 |
| visual_shuffled | 45.97 ± 1.15 | 31.27 ± 1.26 | 10.67 ± 2.15 | 49.79 ± 0.61 | 35.00 ± 2.07 | 11.88 ± 1.84 |

### fusion / NG

| Cấu hình | Head ng-mR@50 | Body ng-mR@50 | Tail ng-mR@50 | Head ng-mR@100 | Body ng-mR@100 | Tail ng-mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 58.28 ± 1.88 | 43.07 ± 2.13 | 20.62 ± 1.37 | 70.95 ± 1.26 | 55.06 ± 1.14 | 29.22 ± 1.78 |
| pair_real | 59.81 ± 0.32 | 44.72 ± 0.33 | 20.21 ± 0.43 | 72.14 ± 0.17 | 56.44 ± 1.34 | 29.67 ± 2.36 |
| pair_shuffled | 59.04 ± 0.67 | 43.69 ± 1.61 | 18.35 ± 0.82 | 71.29 ± 0.90 | 55.65 ± 1.07 | 29.24 ± 1.24 |
| visual_real | 58.37 ± 0.33 | 43.33 ± 1.18 | 19.94 ± 1.22 | 71.12 ± 0.92 | 56.98 ± 1.62 | 29.66 ± 2.15 |
| visual_shuffled | 58.23 ± 1.68 | 43.50 ± 1.77 | 21.16 ± 0.53 | 70.68 ± 1.16 | 56.05 ± 0.93 | 31.31 ± 0.58 |

## Chênh lệch ghép cặp theo seed

Đơn vị điểm phần trăm. Dãy seed theo thứ tự 42 / 123 / 3407. Dấu dương nghĩa là treatment cao hơn control. Đây là thống kê mô tả trên validation, không phải kiểm định ý nghĩa thống kê.

### GC

| Treatment − control | Δ head@50 | Δ body@50 | Δ tail@50 | Δ tail@50 từng seed | Δ tail@100 |
|---|---:|---:|---:|---|---:|
| main: gat − mlp_semantic | +1.66 | +3.62 | +1.41 | +1.22 / +1.89 / +1.13 | +1.15 |
| main: wordnet_kg_on − kg_off | -0.61 | -1.02 | -0.04 | -0.30 / +0.00 / +0.19 | -0.21 |
| main: wordnet_kg_on − wordnet_edges_off | -1.48 | -2.00 | +0.22 | +1.57 / -0.60 / -0.32 | +0.04 |
| main: conceptnet_transe_kg_on − conceptnet_transe_edges_off | -0.63 | -0.63 | +0.50 | +1.67 / +1.00 / -1.17 | +0.96 |
| main: wordnet_kg_on − gat | -0.80 | -1.73 | -0.59 | -0.40 / -0.64 / -0.74 | -0.58 |
| main: conceptnet_transe_kg_on − gat | -0.06 | -0.70 | +0.12 | +0.88 / -0.65 / +0.14 | +0.49 |
| main: conceptnet_rotate_kg_on − gat | -0.35 | -1.16 | -0.66 | -0.79 / -0.94 / -0.26 | -0.79 |
| main: conceptnet_complex_kg_on − gat | -0.19 | -1.49 | +0.33 | +0.21 / -0.72 / +1.49 | +0.31 |
| pair: adjusted_off − ce_off | -0.79 | +10.81 | +6.14 | +4.40 / +9.35 / +4.66 | +6.32 |
| pair: adjusted_on − adjusted_off | -0.51 | +1.96 | -1.28 | -0.16 / -4.28 / +0.59 | -0.77 |
| pair: ce_on − ce_off | +0.30 | +0.05 | +1.23 | +1.02 / +1.85 / +0.83 | +1.23 |
| fusion: pair_real − baseline | +1.04 | -0.19 | -0.48 | +0.78 / -0.51 / -1.72 | -1.11 |
| fusion: pair_real − pair_shuffled | +0.62 | +0.61 | -1.24 | +0.54 / -0.24 / -4.02 | -1.58 |
| fusion: visual_real − baseline | -0.60 | -0.07 | -1.08 | -0.01 / -0.50 / -2.72 | -1.31 |
| fusion: visual_real − visual_shuffled | -0.36 | +1.05 | -2.80 | -1.60 / -0.53 / -6.28 | -2.67 |

### NG

| Treatment − control | Δ head@50 | Δ body@50 | Δ tail@50 | Δ tail@50 từng seed | Δ tail@100 |
|---|---:|---:|---:|---|---:|
| main: gat − mlp_semantic | +2.70 | +3.08 | +0.18 | -1.12 / +3.50 / -1.85 | -1.21 |
| main: wordnet_kg_on − kg_off | -0.88 | +0.35 | -0.75 | -0.70 / -5.47 / +3.93 | -0.23 |
| main: wordnet_kg_on − wordnet_edges_off | -1.47 | -0.67 | -0.17 | +0.45 / +0.98 / -1.93 | +1.44 |
| main: conceptnet_transe_kg_on − conceptnet_transe_edges_off | -1.15 | +0.95 | -3.06 | +1.39 / -5.62 / -4.96 | -0.51 |
| main: wordnet_kg_on − gat | -0.87 | -0.87 | -2.13 | -0.09 / -5.87 / -0.44 | +0.29 |
| main: conceptnet_transe_kg_on − gat | -0.16 | +0.03 | -4.22 | -3.59 / -5.65 / -3.42 | +1.53 |
| main: conceptnet_rotate_kg_on − gat | -0.20 | -0.52 | +0.15 | +1.21 / -0.86 / +0.09 | +0.41 |
| main: conceptnet_complex_kg_on − gat | -0.27 | -0.24 | -0.11 | +1.54 / +1.66 / -3.52 | +1.77 |
| pair: adjusted_off − ce_off | -1.88 | +5.92 | +4.56 | +3.41 / +5.30 / +4.95 | +5.10 |
| pair: adjusted_on − adjusted_off | -0.22 | +0.87 | +1.47 | +1.94 / +0.03 / +2.46 | -0.50 |
| pair: ce_on − ce_off | +1.10 | +0.16 | -0.46 | +7.22 / -4.14 / -4.46 | -0.82 |
| fusion: pair_real − baseline | +1.53 | +1.65 | -0.41 | +0.44 / -1.53 / -0.14 | +0.45 |
| fusion: pair_real − pair_shuffled | +0.76 | +1.03 | +1.86 | +1.24 / +1.93 / +2.41 | +0.43 |
| fusion: visual_real − baseline | +0.09 | +0.26 | -0.68 | -1.16 / -1.77 / +0.88 | +0.44 |
| fusion: visual_real − visual_shuffled | +0.14 | -0.17 | -1.22 | -2.60 / -0.21 / -0.86 | -1.65 |

## Danh sách đủ 50 predicate và tần suất train

| Hạng train | ID | Predicate | Nhóm | Annotation train |
|---:|---:|---|---|---:|
| 1 | 31 | on | head | 118037 |
| 2 | 20 | has | head | 69007 |
| 3 | 48 | wearing | head | 48582 |
| 4 | 30 | of | head | 32770 |
| 5 | 22 | in | head | 24470 |
| 6 | 29 | near | head | 20759 |
| 7 | 8 | behind | head | 13047 |
| 8 | 50 | with | head | 12215 |
| 9 | 21 | holding | head | 11482 |
| 10 | 1 | above | head | 8411 |
| 11 | 40 | sitting on | body | 5355 |
| 12 | 49 | wears | body | 4939 |
| 13 | 43 | under | body | 4732 |
| 14 | 38 | riding | body | 4507 |
| 15 | 23 | in front of | body | 3808 |
| 16 | 41 | standing on | body | 2496 |
| 17 | 6 | at | body | 2109 |
| 18 | 11 | carrying | body | 1705 |
| 19 | 7 | attached to | body | 1586 |
| 20 | 46 | walking on | body | 1322 |
| 21 | 33 | over | body | 1277 |
| 22 | 16 | for | body | 1116 |
| 23 | 25 | looking at | body | 1026 |
| 24 | 47 | watching | body | 907 |
| 25 | 19 | hanging from | body | 807 |
| 26 | 24 | laying on | body | 779 |
| 27 | 14 | eating | body | 702 |
| 28 | 5 | and | body | 679 |
| 29 | 9 | belonging to | body | 652 |
| 30 | 35 | parked on | body | 641 |
| 31 | 44 | using | tail | 580 |
| 32 | 13 | covering | tail | 512 |
| 33 | 10 | between | tail | 511 |
| 34 | 4 | along | tail | 493 |
| 35 | 12 | covered in | tail | 485 |
| 36 | 36 | part of | tail | 435 |
| 37 | 26 | lying on | tail | 369 |
| 38 | 32 | on back of | tail | 343 |
| 39 | 42 | to | tail | 327 |
| 40 | 45 | walking in | tail | 289 |
| 41 | 28 | mounted on | tail | 265 |
| 42 | 2 | across | tail | 263 |
| 43 | 3 | against | tail | 224 |
| 44 | 17 | from | tail | 198 |
| 45 | 18 | growing on | tail | 172 |
| 46 | 34 | painted on | tail | 153 |
| 47 | 37 | playing | tail | 134 |
| 48 | 27 | made of | tail | 128 |
| 49 | 39 | says | tail | 49 |
| 50 | 15 | flying in | tail | 5 |

## Phạm vi và tái lập

Sáu run relation-prior L1/L2 tiếp tục bị loại. Nhóm head/body/tail được khóa từ train trước khi tổng hợp metric; đây vẫn là phân tích hậu kiểm của thí nghiệm cũ, không phải giả thuyết đã đăng ký trước lúc train. Các lớp hiếm có ít annotation/ảnh validation nên cần đọc số support trong CSV cùng với mean/std.

[Nhóm đã khóa](predicate-groups.json) · [Số liệu, từng seed, paired delta và hash nguồn](analysis.json) · [Recall từng predicate, đơn vị fraction 0–1](per-predicate.csv).

```bash
myenv/bin/python -m tools.analyze_level1_long_tail --groups-only
myenv/bin/python -m tools.analyze_level1_long_tail
```

Công cụ kiểm hash report/checkpoint, data hash, support giữa các run, mean/std gốc và tái dựng mR. Chạy lại giữ nguyên output nếu giống hoàn toàn; từ chối ghi đè nếu khác. Dùng `--output-dir` mới cho phân tích phiên bản khác.
