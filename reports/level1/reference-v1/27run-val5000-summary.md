# VG150 PredCls — reference-v1, 27 checkpoint trên validation

Cập nhật: 2026-11-11. Đã evaluate GPU đủ 27/27 checkpoint trên 5.000 ảnh validation.
Ba seed 42/123/3407; bảng là phần trăm, trung bình ± độ lệch chuẩn mẫu.
Checkpoint giữ nguyên từ plan cũ, được chọn bằng metric legacy; không train lại hoặc chọn lại bằng metric mới.

[Protocol mới](../../../docs/evaluation/PredCls.md) · [Summary JSON](27run-val5000-summary.json).

| Phương pháp | R@50 | R@100 | mR@50 | mR@100 |
|---|---:|---:|---:|---:|
| MLP + CLIP | 64.10 ± 0.82 | 66.61 ± 0.35 | 17.21 ± 1.16 | 19.53 ± 1.24 |
| GAT + CLIP | 64.46 ± 0.93 | 66.52 ± 0.95 | 19.55 ± 0.14 | 21.40 ± 0.07 |
| GAT, KG-off | 64.36 ± 0.48 | 66.38 ± 0.53 | 19.01 ± 0.21 | 20.96 ± 0.17 |
| GAT + WordNet TransE | 64.78 ± 0.48 | 66.98 ± 0.40 | 18.46 ± 0.65 | 20.49 ± 0.32 |
| WordNet, edges-off | 65.12 ± 1.13 | 67.15 ± 1.15 | 19.47 ± 1.25 | 21.34 ± 1.16 |
| GAT + ConceptNet TransE | 64.82 ± 0.29 | 66.87 ± 0.30 | 19.31 ± 0.59 | 21.33 ± 0.81 |
| ConceptNet TransE, edges-off | 65.51 ± 0.34 | 67.59 ± 0.34 | 19.49 ± 0.94 | 21.32 ± 0.84 |
| GAT + ConceptNet RotatE | 64.71 ± 0.68 | 66.79 ± 0.72 | 18.75 ± 0.53 | 20.41 ± 0.45 |
| GAT + ConceptNet ComplEx | 64.93 ± 0.35 | 66.99 ± 0.44 | 19.05 ± 0.23 | 21.02 ± 0.28 |

GAT tăng mR@50 khoảng 2,34 điểm phần trăm so với MLP. Bốn nhánh KG-on đều thấp hơn GAT về mR@50 trung bình, dù R@50 cao hơn nhẹ. Chưa chứng minh lợi ích KG ổn định.

Đây là validation, không phải test mới. 12 run pair/loss và 21 run fusion chưa được chấm lại trong đợt này; số khoảng 25–26% trước đó vẫn thuộc protocol legacy.

Kiểm tra: 212 tests PASS; differential check với code tham chiếu đúng commit trên 100 trường hợp tổng hợp và 16 ảnh validation thật PASS; 27/27 report hash, evaluator source hash và support theo lớp khớp. CPU smoke 16 ảnh cho MLP/GAT/WordNet PASS. Không dùng subset smoke để kết luận chất lượng.

Môi trường full evaluation: NVIDIA GeForce RTX 5090, PyTorch 2.7.0+cu128, CUDA 12.8, batch 128. Inference trên GPU; matching/aggregation trên CPU. mR lấy trung bình cố định 50 lớp; lớp không có support đóng góp 0.

Tương thích đã kiểm chứng ở mức evaluator trên cùng boxes/labels/logits. Không khẳng định mọi khác biệt preprocessing và huấn luyện với từng paper đã được loại bỏ.
