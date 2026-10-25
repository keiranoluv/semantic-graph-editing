# VG150 PredCls — 24 run CLIP, trừ ComplEx (bảng tạm thời)

Phạm vi: 21 report đánh giá độc lập đã có trong
[bảng cũ](clip-all-21run-val5000-summary.md), cộng 3 checkpoint RotatE KG-on.
Cả 24 run dùng 5.000 ảnh validation, 49 predicate có GT, cùng split VG150,
cache CLIP và các siêu tham số PredCls chung. Ba dòng RotatE lấy metric
`validation` trong `best.pt` ở epoch được chọn; **chưa có report evaluate độc
lập**. Vì vậy bảng này chưa phải bản chốt để báo cáo. Không gồm bất kỳ run
ComplEx nào, annotation-only hoặc VG150 test.
Hash mã nguồn khác giữa các đợt train do thêm nhánh KG mới; hash dữ liệu,
cache CLIP và các tham số mô hình/chạy chung đã được đối chiếu.

Các số dưới đây là **phần trăm**; `±` là sample standard deviation qua seeds
42, 123, 3407. Chọn checkpoint theo validation mR@50.

## Trung bình ba seed

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
| --- | ---: | ---: | ---: | ---: |
| MLP + CLIP | 62,91 ± 0,77 | 17,37 ± 1,12 | 65,52 ± 0,31 | 19,78 ± 0,97 |
| GAT + CLIP | 63,24 ± 0,97 | **19,98 ± 0,47** | 65,40 ± 1,01 | **22,04 ± 0,38** |
| GAT + CLIP, KG-off | 63,16 ± 0,49 | 19,63 ± 0,25 | 65,26 ± 0,52 | 21,87 ± 0,40 |
| GAT + CLIP + WordNet | 63,62 ± 0,49 | 18,85 ± 0,40 | 65,93 ± 0,46 | 21,16 ± 0,35 |
| WordNet, edges-off | 63,85 ± 1,22 | 19,67 ± 1,16 | 65,98 ± 1,26 | 21,70 ± 1,11 |
| GAT + CLIP + ConceptNet TransE | 63,67 ± 0,34 | 19,66 ± 0,66 | 65,84 ± 0,33 | 21,84 ± 0,75 |
| ConceptNet TransE, edges-off | **64,32 ± 0,40** | 19,86 ± 0,85 | **66,50 ± 0,37** | 21,87 ± 0,72 |
| GAT + CLIP + ConceptNet RotatE¹ | 63,54 ± 0,77 | 19,08 ± 0,70 | 65,75 ± 0,81 | 21,07 ± 0,59 |

¹ RotatE dùng metric lưu trong checkpoint; các điều kiện khác dùng report
evaluate độc lập. Chênh lệch mR@50 trung bình RotatE so với GAT, KG-off và
ConceptNet TransE lần lượt là −0,90, −0,55 và −0,58 điểm phần trăm.

## Từng run

| Điều kiện | Seed | Best epoch | R@50 | mR@50 | R@100 | mR@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP + CLIP | 42 | 81 | 63,05 | 18,18 | 65,44 | 20,77 |
| MLP + CLIP | 123 | 20 | 62,08 | 16,09 | 65,27 | 18,83 |
| MLP + CLIP | 3407 | 78 | 63,59 | 17,84 | 65,87 | 19,74 |
| GAT + CLIP | 42 | 45 | 63,83 | 20,35 | 66,02 | 22,32 |
| GAT + CLIP | 123 | 45 | 63,77 | 19,44 | 65,95 | 21,61 |
| GAT + CLIP | 3407 | 85 | 62,12 | 20,14 | 64,24 | 22,19 |
| GAT + CLIP, KG-off | 42 | 55 | 62,89 | 19,89 | 65,02 | 21,84 |
| GAT + CLIP, KG-off | 123 | 43 | 63,73 | 19,39 | 65,87 | 21,48 |
| GAT + CLIP, KG-off | 3407 | 60 | 62,88 | 19,61 | 64,91 | 22,28 |
| GAT + CLIP + WordNet | 42 | 26 | 64,19 | 19,28 | 66,36 | 21,34 |
| GAT + CLIP + WordNet | 123 | 48 | 63,31 | 18,79 | 65,45 | 20,75 |
| GAT + CLIP + WordNet | 3407 | 27 | 63,37 | 18,48 | 65,98 | 21,38 |
| WordNet, edges-off | 42 | 26 | 64,68 | 18,70 | 66,77 | 20,80 |
| WordNet, edges-off | 123 | 108 | 62,44 | 20,96 | 64,53 | 22,93 |
| WordNet, edges-off | 3407 | 41 | 64,43 | 19,37 | 66,65 | 21,36 |
| GAT + CLIP + ConceptNet TransE | 42 | 45 | 63,61 | 20,42 | 65,95 | 22,69 |
| GAT + CLIP + ConceptNet TransE | 123 | 56 | 63,36 | 19,28 | 65,47 | 21,28 |
| GAT + CLIP + ConceptNet TransE | 3407 | 42 | 64,03 | 19,29 | 66,11 | 21,55 |
| ConceptNet TransE, edges-off | 42 | 40 | 64,28 | 19,58 | 66,40 | 21,89 |
| ConceptNet TransE, edges-off | 123 | 35 | 63,94 | 19,18 | 66,20 | 21,15 |
| ConceptNet TransE, edges-off | 3407 | 29 | 64,73 | 20,82 | 66,91 | 22,58 |
| GAT + CLIP + ConceptNet RotatE¹ | 42 | 60 | 62,66 | 19,63 | 64,82 | 21,66 |
| GAT + CLIP + ConceptNet RotatE¹ | 123 | 38 | 64,12 | 19,32 | 66,29 | 21,07 |
| GAT + CLIP + ConceptNet RotatE¹ | 3407 | 32 | 63,84 | 18,30 | 66,14 | 20,47 |

Kết luận hiện tại theo mR@50: GAT thuần cao nhất trong 8 điều kiện. KG-on
WordNet, ConceptNet TransE và ConceptNet RotatE đều thấp hơn GAT trung bình;
ConceptNet TransE gần KG-off nhất. RotatE seed 123 nhỉnh hơn ConceptNet
TransE seed 123 khoảng 0,04 điểm, còn hai seed kia thấp hơn. R@50 và mR@50
có thể đi khác hướng; chọn mô hình cuối phải xem cả hai và chạy evaluate
RotatE độc lập trước khi khóa bảng.
