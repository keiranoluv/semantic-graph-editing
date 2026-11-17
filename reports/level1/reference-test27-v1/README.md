# VG150 PredCls — kết quả test reference-v1 của 27 checkpoint

## Phạm vi và xác minh

Đã hoàn tất **9 cấu hình × 3 seed (42, 123, 3407)**, mỗi lượt trên toàn bộ
**26.446 ảnh test**, đủ 50 predicate có ground truth. Checkpoint thuộc ma trận
gốc đã khóa; lần này chỉ chạy inference bằng evaluator reference-v1 GC.
Batch inference 128, một checkpoint mỗi thời điểm. Không train/chọn lại epoch.

Audit CPU kiểm hash plan, 27 checkpoint, 27 report, evaluator và cache; kiểm
split/số ảnh/seed và tính lại mean/sample standard deviation từ từng report.
Kết quả **PASS** trong [audit.json](audit.json). Chạy audit không inference GPU.

Protocol: `PredCls_reference_v1_iou50_image_macro_per_predicate_50`.
GT box và nhãn object được cho trước; một predicate foreground mỗi cặp có hướng;
matching theo lớp và IoU ≥ 0,5 của cả hai hộp; giữ annotation GT trùng;
mR trung bình theo ảnh cho mỗi predicate rồi trung bình đủ 50 lớp.
Checkpoint được chọn bằng metric legacy trong huấn luyện. Test đã được xem ở
lượt legacy trước đây: đây là tái chấm checkpoint cố định bằng metric mới,
không phải một tập holdout mới cho các thiết kế phát triển sau đó.

## Bảng kết quả chính

Đơn vị %, mean ± sample std qua ba seed. Bảng chỉ dùng reference-v1 test.

| Cấu hình | R@50 | R@100 | mR@50 | mR@100 |
|---|---:|---:|---:|---:|
| MLP + CLIP | 63,56 ± 0,77 | 66,40 ± 0,29 | 16,95 ± 1,52 | 19,52 ± 1,42 |
| GAT + CLIP | 63,87 ± 0,73 | 66,21 ± 0,70 | **18,61 ± 0,25** | **20,95 ± 0,21** |
| GAT KG-off | 63,89 ± 0,38 | 66,24 ± 0,30 | 18,44 ± 0,29 | 20,72 ± 0,43 |
| WordNet TransE | 64,15 ± 0,29 | 66,69 ± 0,34 | 17,84 ± 0,09 | 20,34 ± 0,08 |
| WordNet edges-off | 64,11 ± 1,15 | 66,46 ± 1,13 | 18,41 ± 0,90 | 20,72 ± 0,76 |
| ConceptNet TransE | 64,10 ± 0,27 | 66,45 ± 0,28 | 18,39 ± 0,29 | 20,71 ± 0,28 |
| ConceptNet TransE edges-off | **64,59 ± 0,19** | **66,97 ± 0,16** | 18,15 ± 0,28 | 20,47 ± 0,32 |
| ConceptNet RotatE | 64,21 ± 0,37 | 66,60 ± 0,46 | 18,57 ± 0,42 | 20,93 ± 0,35 |
| ConceptNet ComplEx | 64,37 ± 0,18 | 66,79 ± 0,25 | 18,33 ± 0,24 | 20,75 ± 0,18 |

## So sánh và ablation

Delta tính từ số chưa làm tròn, đơn vị điểm phần trăm (pp).

| Treatment − control | ΔR@50 | ΔmR@50 | ΔmR@50 từng seed 42 / 123 / 3407 |
|---|---:|---:|---|
| GAT − MLP | +0,312 | +1,654 | +0,970 / +3,131 / +0,861 |
| WordNet − KG-off | +0,261 | −0,604 | −0,827 / −0,247 / −0,739 |
| WordNet − WordNet edges-off | +0,044 | −0,573 | +0,376 / −1,422 / −0,673 |
| ConceptNet TransE − edges-off | −0,491 | +0,240 | +0,018 / +0,386 / +0,315 |
| WordNet − GAT | +0,276 | −0,767 | −0,778 / −0,444 / −1,080 |
| ConceptNet TransE − GAT | +0,227 | −0,216 | −0,456 / −0,097 / −0,096 |
| RotatE − GAT | +0,342 | −0,035 | +0,375 / +0,089 / −0,569 |
| ComplEx − GAT | +0,500 | −0,271 | −0,072 / −0,043 / −0,698 |

**Message passing:** GAT cải thiện mR@50 so với MLP ở cả ba seed, đồng thời
mR@100 tăng 1,427 pp. R@100 giảm 0,191 pp, nên lợi ích nằm ở recall cân bằng
theo lớp, không phải mọi chỉ số đều tăng. Trong nhánh ConceptNet TransE,
giữ message passing tăng mR@50 0,240 pp so với edges-off nhưng giảm R@50.
Nhánh WordNet cho chiều tác động khác; không suy mọi kiến trúc GNN đều tốt hơn
đối chứng bỏ cạnh.

**Tri thức nền:** cả bốn KG-on có R@50 cao hơn GAT nhưng mR@50 thấp hơn.
WordNet kém KG-off cùng kiến trúc ở cả ba seed là đối chứng trực tiếp cho thấy
embedding WordNet hiện tại chưa cải thiện balanced recall. KG-off dùng bundle
WordNet, không coi đây là đối chứng cùng kiến trúc hoàn toàn cho mọi ConceptNet.
RotatE gần hòa GAT; hai seed tăng và một seed giảm, chưa có lợi ích ổn định.
Ba seed cùng dấu chỉ là quan sát độ ổn định, không phải kiểm định ý nghĩa thống kê.

## Validation và test cùng evaluator

Đã đối chiếu hash checkpoint/protocol với 27 report validation reference-v1.

| Cấu hình | Test − validation mR@50 (pp) |
|---|---:|
| MLP | −0,257 |
| GAT | −0,947 |
| KG-off | −0,566 |
| WordNet | −0,626 |
| WordNet edges-off | −1,062 |
| ConceptNet TransE | −0,921 |
| ConceptNet edges-off | −1,338 |
| RotatE | −0,183 |
| ComplEx | −0,715 |

Cả chín cấu hình giảm mR@50 khi chuyển sang test. Đây là kết quả mô tả trên
hai split; dữ liệu hiện tại không tách riêng tác động chọn checkpoint và khác
biệt phân bố. Không dùng chênh lệch giữa legacy test và reference test để tuyên bố
mô hình cải thiện: trọng số không đổi, cách matching và tổng hợp metric đã đổi.

## Sử dụng trong báo cáo đồ án

Dùng bảng test reference-v1 này làm bảng chính của Nhiệm vụ 1. Kết luận phù hợp:
GAT cải thiện balanced recall so với MLP; KG làm tăng nhẹ Recall tổng thể nhưng
chưa tăng mean Recall so với GAT. Các bảng adjusted loss, pair/visual fusion,
shuffled-KG và zero-shot trước đó vẫn là validation, không gắn nhãn test cho chúng.
Lượt này không đo lại các nhánh đó và không suy kết luận long-tail theo nhóm
từ mR toàn bộ. Muốn kết luận riêng head/body/tail cần tổng hợp per-predicate
theo nhóm lớp đã khóa từ train.

## Artifacts và tái lập

- [summary.json](summary.json): mean/std, protocol và SHA256 từng report.
- 27 file `*-test26446.json`: metric, per-predicate, checkpoint/config/evaluator provenance.
- [per-seed.csv](per-seed.csv): bốn metric từng seed, đơn vị %.
- [comparisons.json](comparisons.json): paired delta và test−validation, đơn vị pp.
- [audit.json](audit.json): kết quả audit và hash mã phân tích.
- Plan gốc: `reports/level1/clip-all-27run-test-plan.json`.
- Launcher: commit `cd64cf8`, `tools/level1_reference_test27_v1.py`.

Tính lại thống kê và kiểm chứng, chỉ CPU, từ gốc repository:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.analyze_reference_test27_v1
```

Lệnh từ chối thay artifact phân tích nếu nội dung khác; không sửa report inference,
checkpoint, manifest, notebook hoặc kết quả legacy.
