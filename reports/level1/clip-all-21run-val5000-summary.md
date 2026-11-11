# VG150 PredCls — toàn bộ 21 run có CLIP (validation 5.000 ảnh)

Nguồn: 21 report `val5000.json` và `best.pt` tương ứng. Mọi report cùng `split=val`, cùng 5.000 ảnh, 49 predicate có GT, `PredCls_graph_constrained_top1_per_pair`, hash dữ liệu/cache CLIP và siêu tham số chung. Mọi metric trong report khớp tuyệt đối với `validation` lưu trong `best.pt`. Hai đợt WordNet/ConceptNet có `code_sha256` khác do thay đổi mã giữa các đợt; các tham số chung và dữ liệu đã được đối chiếu.

Giá trị trong bảng là **phần trăm**. `±` là độ lệch chuẩn mẫu qua ba seed 42, 123, 3407. Đây là validation dùng chọn checkpoint, chưa phải held-out test.

## Trung bình ba seed

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
| --- | ---: | ---: | ---: | ---: |
| MLP + CLIP | 62.91 ± 0.77 | 17.37 ± 1.12 | 65.52 ± 0.31 | 19.78 ± 0.97 |
| GAT + CLIP | 63.24 ± 0.97 | 19.98 ± 0.47 | 65.40 ± 1.01 | 22.04 ± 0.38 |
| GAT + CLIP, KG-off | 63.16 ± 0.49 | 19.63 ± 0.25 | 65.26 ± 0.52 | 21.87 ± 0.40 |
| GAT + CLIP + WordNet | 63.62 ± 0.49 | 18.85 ± 0.40 | 65.93 ± 0.46 | 21.16 ± 0.35 |
| WordNet, image edges off | 63.85 ± 1.22 | 19.67 ± 1.16 | 65.98 ± 1.26 | 21.70 ± 1.11 |
| GAT + CLIP + ConceptNet | 63.67 ± 0.34 | 19.66 ± 0.66 | 65.84 ± 0.33 | 21.84 ± 0.75 |
| ConceptNet, image edges off | 64.32 ± 0.40 | 19.86 ± 0.85 | 66.50 ± 0.37 | 21.87 ± 0.72 |

## Từng run

| Điều kiện | Seed | Best epoch | R@50 | mR@50 | R@100 | mR@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP + CLIP | 42 | 81 | 63.05 | 18.18 | 65.44 | 20.77 |
| MLP + CLIP | 123 | 20 | 62.08 | 16.09 | 65.27 | 18.83 |
| MLP + CLIP | 3407 | 78 | 63.59 | 17.84 | 65.87 | 19.74 |
| GAT + CLIP | 42 | 45 | 63.83 | 20.35 | 66.02 | 22.32 |
| GAT + CLIP | 123 | 45 | 63.77 | 19.44 | 65.95 | 21.61 |
| GAT + CLIP | 3407 | 85 | 62.12 | 20.14 | 64.24 | 22.19 |
| GAT + CLIP, KG-off | 42 | 55 | 62.89 | 19.89 | 65.02 | 21.84 |
| GAT + CLIP, KG-off | 123 | 43 | 63.73 | 19.39 | 65.87 | 21.48 |
| GAT + CLIP, KG-off | 3407 | 60 | 62.88 | 19.61 | 64.91 | 22.28 |
| GAT + CLIP + WordNet | 42 | 26 | 64.19 | 19.28 | 66.36 | 21.34 |
| GAT + CLIP + WordNet | 123 | 48 | 63.31 | 18.79 | 65.45 | 20.75 |
| GAT + CLIP + WordNet | 3407 | 27 | 63.37 | 18.48 | 65.98 | 21.38 |
| WordNet, image edges off | 42 | 26 | 64.68 | 18.70 | 66.77 | 20.80 |
| WordNet, image edges off | 123 | 108 | 62.44 | 20.96 | 64.53 | 22.93 |
| WordNet, image edges off | 3407 | 41 | 64.43 | 19.37 | 66.65 | 21.36 |
| GAT + CLIP + ConceptNet | 42 | 45 | 63.61 | 20.42 | 65.95 | 22.69 |
| GAT + CLIP + ConceptNet | 123 | 56 | 63.36 | 19.28 | 65.47 | 21.28 |
| GAT + CLIP + ConceptNet | 3407 | 42 | 64.03 | 19.29 | 66.11 | 21.55 |
| ConceptNet, image edges off | 42 | 40 | 64.28 | 19.58 | 66.40 | 21.89 |
| ConceptNet, image edges off | 123 | 35 | 63.94 | 19.18 | 66.20 | 21.15 |
| ConceptNet, image edges off | 3407 | 29 | 64.73 | 20.82 | 66.91 | 22.58 |

## Chênh lệch chính (điểm phần trăm)

| So sánh | ΔR@50 | ΔmR@50 | ΔR@100 | ΔmR@100 |
| --- | ---: | ---: | ---: | ---: |
| ConceptNet KG-on − WordNet KG-on | +0.04 | +0.82 | -0.09 | +0.68 |
| ConceptNet KG-on − KG-off | +0.50 | +0.04 | +0.58 | -0.02 |
| ConceptNet KG-on − GAT | +0.42 | -0.31 | +0.44 | -0.20 |
| ConceptNet edges off − WordNet edges off | +0.47 | +0.19 | +0.52 | +0.18 |
| ConceptNet edges off − ConceptNet KG-on | +0.65 | +0.20 | +0.66 | +0.03 |

`KG-off` dùng kiến trúc GAT+KG nhưng cả vector KG và valid bit bằng 0, nên ba run CLIP cũ được dùng lại làm đối chứng ConceptNet. `image edges off` vẫn dùng vector KG, chỉ tắt truyền thông điệp giữa các object trong ảnh (còn self-loop).

Nhận định theo mR@50: ConceptNet KG-on cao hơn WordNet KG-on ở cả ba seed và trung bình +0,82 điểm; so với KG-off chỉ +0,04 điểm và chỉ thắng một seed; thấp hơn GAT thuần 0,31 điểm. Vì vậy chưa có bằng chứng vững rằng KG cải thiện baseline không KG. R@50 của ConceptNet KG-on tăng so với GAT/KG-off, cho thấy trade-off giữa recall tổng thể và recall trung bình predicate.

Nguồn tổng hợp gốc: `clip-v1-matched-3seed-val5000-summary.json` và `clip-conceptnet-v1-matched-3seed-val5000-summary.json`. Không trộn bảng này với annotation-only hoặc test split.
