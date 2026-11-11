# So sánh graph constraint và no graph constraint — Mức 1

54 checkpoint cố định, cùng 5.000 ảnh validation, seed 42/123/3407. Đơn vị %, mean ± sample std. Không train/chọn lại checkpoint; metric chọn checkpoint vẫn là legacy.

GC: mỗi cặp có hướng giữ một predicate. NG: tất cả predicate foreground tham gia top-K toàn ảnh. Matching và cách trung bình recall giống nhau. Hai quan hệ cùng cặp chiếm hai vị trí; NG không bảo đảm tăng recall.

## main — 27 lượt chạy

| Cấu hình | R@50 GC | ng-R@50 | mR@50 GC | ng-mR@50 | ng-R@100 | ng-mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| mlp_semantic | 64.10 ± 0.82 | 76.53 ± 1.92 | 17.21 ± 1.16 | 31.37 ± 2.54 | 84.71 ± 1.45 | 42.76 ± 3.26 |
| gat | 64.46 ± 0.93 | 77.87 ± 0.48 | 19.55 ± 0.14 | 33.21 ± 0.70 | 85.65 ± 0.63 | 43.86 ± 2.15 |
| kg_off | 64.36 ± 0.48 | 77.95 ± 0.25 | 19.01 ± 0.21 | 32.18 ± 1.23 | 85.66 ± 0.36 | 43.99 ± 1.28 |
| wordnet_kg_on | 64.78 ± 0.48 | 77.69 ± 0.69 | 18.46 ± 0.65 | 31.84 ± 1.25 | 85.62 ± 0.43 | 43.87 ± 1.08 |
| wordnet_edges_off | 65.12 ± 1.13 | 78.19 ± 1.05 | 19.47 ± 1.25 | 32.47 ± 1.83 | 86.01 ± 0.71 | 43.84 ± 2.19 |
| conceptnet_transe_kg_on | 64.82 ± 0.29 | 78.14 ± 0.36 | 19.31 ± 0.59 | 31.51 ± 0.94 | 85.90 ± 0.24 | 44.59 ± 1.05 |
| conceptnet_transe_edges_off | 65.51 ± 0.34 | 78.71 ± 0.18 | 19.49 ± 0.94 | 32.58 ± 1.32 | 86.29 ± 0.13 | 45.43 ± 0.63 |
| conceptnet_rotate_kg_on | 64.71 ± 0.68 | 77.97 ± 0.29 | 18.75 ± 0.53 | 33.02 ± 0.76 | 85.77 ± 0.19 | 44.30 ± 0.84 |
| conceptnet_complex_kg_on | 64.93 ± 0.35 | 78.12 ± 0.15 | 19.05 ± 0.23 | 33.02 ± 1.90 | 85.93 ± 0.22 | 45.02 ± 0.52 |

## pair — 12 lượt chạy

| Cấu hình | R@50 GC | ng-R@50 | mR@50 GC | ng-mR@50 | ng-R@100 | ng-mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| ce_off | 65.16 ± 0.10 | 78.01 ± 0.11 | 19.18 ± 0.57 | 32.25 ± 0.47 | 85.88 ± 0.18 | 44.46 ± 0.42 |
| ce_on | 64.05 ± 0.15 | 77.80 ± 0.22 | 19.75 ± 0.99 | 32.35 ± 1.49 | 85.71 ± 0.24 | 44.57 ± 0.23 |
| adjusted_off | 61.97 ± 0.12 | 75.39 ± 0.23 | 25.80 ± 1.10 | 36.06 ± 1.18 | 84.06 ± 0.17 | 47.93 ± 0.34 |
| adjusted_on | 61.97 ± 0.42 | 75.19 ± 0.28 | 25.97 ± 0.61 | 36.96 ± 1.31 | 84.05 ± 0.19 | 48.27 ± 1.27 |

## fusion — 15 lượt chạy

| Cấu hình | R@50 GC | ng-R@50 | mR@50 GC | ng-mR@50 | ng-R@100 | ng-mR@100 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 61.96 ± 0.67 | 75.00 ± 0.91 | 25.78 ± 0.99 | 37.13 ± 1.62 | 83.76 ± 0.45 | 47.90 ± 1.06 |
| pair_real | 62.64 ± 0.41 | 75.91 ± 0.12 | 25.72 ± 0.30 | 37.93 ± 0.10 | 84.45 ± 0.22 | 48.87 ± 0.58 |
| pair_shuffled | 62.40 ± 0.13 | 75.65 ± 0.28 | 25.85 ± 0.09 | 36.62 ± 1.09 | 84.18 ± 0.26 | 48.21 ± 0.25 |
| visual_real | 62.81 ± 0.09 | 75.82 ± 0.45 | 25.20 ± 0.30 | 36.98 ± 0.97 | 84.40 ± 0.47 | 48.88 ± 0.31 |
| visual_shuffled | 62.82 ± 0.49 | 75.58 ± 0.98 | 25.97 ± 0.91 | 37.51 ± 1.16 | 84.17 ± 0.57 | 49.08 ± 0.62 |

## Chênh lệch giữa các mô hình trong NG

Đơn vị điểm phần trăm. Paired delta theo seed 42 / 123 / 3407; đây là thống kê mô tả, không phải kiểm định ý nghĩa.

| So sánh | Δ ng-R@50 trung bình | Δ ng-mR@50 trung bình | Δ ng-mR@50 từng seed |
|---|---:|---:|---|
| main: gat − mlp_semantic | +1.34 | +1.84 | +0.75 / +5.14 / -0.36 |
| main: wordnet_kg_on − gat | -0.18 | -1.38 | -0.42 / -2.73 / -0.97 |
| main: conceptnet_transe_kg_on − gat | +0.27 | -1.71 | -1.81 / -3.13 / -0.18 |
| main: conceptnet_rotate_kg_on − gat | +0.09 | -0.19 | +0.12 / -0.55 / -0.15 |
| main: conceptnet_complex_kg_on − gat | +0.25 | -0.20 | +0.39 / +0.60 / -1.58 |
| pair: adjusted_off − ce_off | -2.62 | +3.81 | +3.32 / +4.78 / +3.34 |
| pair: adjusted_on − adjusted_off | -0.20 | +0.90 | +0.60 / -0.33 / +2.41 |
| pair: ce_on − ce_off | -0.21 | +0.10 | +2.35 / -0.79 / -1.25 |
| fusion: pair_real − baseline | +0.92 | +0.80 | +1.91 / -0.95 / +1.43 |
| fusion: pair_real − pair_shuffled | +0.27 | +1.31 | +0.40 / +1.08 / +2.45 |
| fusion: visual_real − baseline | +0.82 | -0.15 | -0.07 / -1.78 / +1.41 |
| fusion: visual_real − visual_shuffled | +0.24 | -0.53 | -2.46 / +1.01 / -0.13 |

Sáu run relation-prior L1/L2 vẫn bị loại. Đây là validation của checkpoint đã chọn theo legacy, không phải test độc lập. Không so ng-mR của mô hình này với mR có graph constraint của mô hình khác.
