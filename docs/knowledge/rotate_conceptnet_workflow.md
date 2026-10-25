# Thử RotatE với ConceptNet reviewed ở mức 1

Mục tiêu: thay **cách nhúng KG** từ TransE sang RotatE, giữ nguyên ConceptNet
reviewed v3, mapping 138/150 class, cache CLIP, split VG150, mô hình GAT+KG và
siêu tham số PredCls. Đây là thử nghiệm trên validation, chưa dùng VG150 test.

RotatE dùng [`torch_geometric.nn.kge.RotatE`](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.kge.RotatE.html)
đã có trong `requirements.txt`. Trainer **không gọi** `RotatE.loss()` của PyG:
hàm đó tự sinh negative ngẫu nhiên. Bản v2 dùng embedding/pha quan hệ của PyG,
điểm phức L1 và [self-adversarial logistic loss từ mã gốc RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py).
Mỗi positive có 32 negative được lọc trên triple train; negative khó được
trọng số lớn hơn, không truyền gradient qua trọng số. Validation
ranking chỉ lọc train+val; KG test ranking chỉ thực hiện sau chọn checkpoint
và lọc cả ba split. KG test ở đây là link prediction, không phải VG150 test.

`--dim 64` nghĩa là 32 chiều phức: khi export, ghép real/imag thành vector
thực 64D, chuẩn hóa L2 từng entity, rồi đóng băng trong GAT+KG. Class reject
và background giữ vector 0, valid mask false. Khác biệt kết quả PredCls vì
vậy kiểm tra cả phương pháp nhúng lẫn objective huấn luyện KG RotatE so với
TransE; không quy chênh lệch cho riêng phép quay. Hai nhánh dùng cùng split KG
theo seed 42 và cùng 512 triple validation/test được lấy mẫu.

**Bản v1 đã thất bại:** 21 epochs, best tại epoch 1 với validation MRR
0,00455 so với TransE 0,01567. Train loss giảm 0,991→0,041 trong khi
validation MRR tụt về khoảng 0,0005; Hits@1 luôn bằng 0. v1 dùng một negative
và margin hinge loss với score L2 của PyG. Giữ artifact
`runs/rotate/conceptnet-toan-v3-seed42/` để audit; không export v1 hay chạy
PredCls từ nó. Bản v2 là một thử nghiệm sửa phương pháp train, chưa được
chứng minh có metric tốt hơn.

## 1. Preflight và train KG

Làm việc tại gốc repository. CPU preflight chỉ in manifest/số triple:

```bash
myenv/bin/python -m scripts.train_rotate_v2 \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --mode reviewed --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --split-policy transductive --val-ratio 0.08 --test-ratio 0.08 \
  --eval-max-triples 512 --output-dir runs/rotate/conceptnet-toan-v3-selfadv-v2-seed42 \
  --dim 64 --margin 6 --lr 0.0001 --negative-count 32 --adversarial-temperature 1 \
  --epochs 200 --patience 40 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cpu --wandb-mode disabled --dry-run
```

**Toan chạy GPU train KG** (một run). Thư mục output phải chưa tồn tại:

```bash
myenv/bin/python -u -m scripts.train_rotate_v2 \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --mode reviewed --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --split-policy transductive --val-ratio 0.08 --test-ratio 0.08 \
  --eval-max-triples 512 --output-dir runs/rotate/conceptnet-toan-v3-selfadv-v2-seed42 \
  --dim 64 --margin 6 --lr 0.0001 --negative-count 32 --adversarial-temperature 1 \
  --epochs 200 --patience 40 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cuda --wandb-mode online
```

Đầu ra cần báo lại: `best.pt`, `metrics.json`, best epoch và validation MRR.
Không diễn giải KG MRR như PredCls mR@50. Đường dẫn mới giữ nguyên cả run
TransE và RotatE v1. Chỉ export khi validation MRR cho thấy có tín hiệu; MRR
thấp hơn TransE không tự động suy ra PredCls kém, nhưng cần xem xét trước khi
tốn thêm ba run GPU.

## 2. Export C.2 trên CPU

Sau khi lệnh train KG kết thúc, Codex hoặc teammate chạy:

```bash
myenv/bin/python -m scripts.export_conceptnet_rotate \
  --checkpoint runs/rotate/conceptnet-toan-v3-selfadv-v2-seed42/best.pt \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --output-dir data/knowledge/c2-conceptnet-rotate-toan-v2
```

Script kiểm hash KG/review/checkpoint, thứ tự class ID và 138 valid bit. Nó
từ chối checkpoint TransE/RotatE v1 hoặc output đã tồn tại.

## 3. Ba run PredCls KG-on

Xem lệnh trước mà không chạy GPU:

```bash
myenv/bin/python -m src.level1_rotate_matrix train --max-parallel 3 --dry-run
```

**Toan chạy GPU train và evaluate** sau khi export thành công:

```bash
myenv/bin/python -m src.level1_rotate_matrix train --max-parallel 3
myenv/bin/python -m src.level1_rotate_matrix evaluate --device cuda --max-parallel 1
```

Launcher chạy `kg_on` seeds 42/123/3407; có thể chọn một ô bằng
`--condition kg_on --seed 42`. Cả ba dùng 57.723 ảnh train, 5.000 ảnh
validation, batch 512, tối đa 200 epochs, patience 15 và cùng CLIP cache.
Evaluate ghi ba JSON và một summary riêng trong `reports/level1/`.

Đối chiếu mR@50, R@50 từng seed và mean±sample SD với ConceptNet+TransE
KG-on (19,66% mR@50), KG-off (19,63%) và GAT thuần (19,98%) trong
[`clip-all-21run-val5000-summary.md`](../../reports/level1/clip-all-21run-val5000-summary.md).
Nếu kết quả đáng theo tiếp, chạy thêm ba run `edges_off` bằng
`--condition edges_off`, rồi đánh giá tương ứng; chúng phải train lại vì
vẫn dùng vector RotatE. MLP/GAT/KG-off hiện có được tái sử dụng.

Nguồn tải ConceptNet 5.7 và lệnh tái dựng KG cho teammate:
[`b2_conceptnet_workflow.md`](b2_conceptnet_workflow.md).
