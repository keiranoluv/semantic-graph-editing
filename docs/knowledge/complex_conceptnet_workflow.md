# Thử ComplEx với ConceptNet reviewed ở mức 1

ComplEx là một lựa chọn embedding KG được nêu trong đề bài. Nhánh này dùng
chính ConceptNet reviewed v3 (138/150 class được duyệt), cùng split KG seed 42
và cùng cache CLIP, cấu hình GAT+KG, ba seed PredCls 42/123/3407. MLP, GAT
thuần và KG-off trước được dùng lại. Không có bridge WordNet↔ConceptNet.

Trainer dùng [`torch_geometric.nn.kge.ComplEx`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.ComplEx.html)
để tính điểm Hermitian, logistic loss với 32 negative/triple và L2
regularization theo [công bố ComplEx](https://proceedings.mlr.press/v48/trouillon16.pdf).
Negative sampler của project loại triple **train** đúng; không gọi
`ComplEx.loss()` của PyG vì hàm đó tự lấy negative không lọc. Ranking
validation loại train+val; KG test chỉ tính sau chọn checkpoint và loại cả
train+val+test. Kết quả KG link prediction không phải VG150 PredCls.

`--dim 64` là 32 chiều phức; export ghép real/imag thành 64 số thực, chuẩn hóa
L2 mỗi entity, đóng băng ở GAT+KG. Class reject/background có vector 0 và
mask false. So sánh với TransE/RotatE là so **pipeline nhúng KG hoàn chỉnh**:
score, loss và siêu tham số KG khác nhau; không quy chênh lệch cho riêng phép
nhân phức. Hai bước PredCls và CLIP giữ cùng protocol.

## 1. KG train: Toan chạy GPU

Đứng tại gốc repository. CPU preflight:

```bash
myenv/bin/python -m scripts.train_complex \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --mode reviewed --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --split-policy transductive --val-ratio 0.08 --test-ratio 0.08 \
  --eval-max-triples 512 --output-dir runs/complex/conceptnet-toan-v3-seed42 \
  --dim 64 --lr 0.001 --negative-count 32 --l2-regularization 0.001 \
  --epochs 200 --patience 40 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cpu --wandb-mode disabled --dry-run
```

**Sau khi commit code, Toan chạy GPU**:

```bash
myenv/bin/python -u -m scripts.train_complex \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --mode reviewed --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --split-policy transductive --val-ratio 0.08 --test-ratio 0.08 \
  --eval-max-triples 512 --output-dir runs/complex/conceptnet-toan-v3-seed42 \
  --dim 64 --lr 0.001 --negative-count 32 --l2-regularization 0.001 \
  --epochs 200 --patience 40 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cuda --wandb-mode online
```

Đầu ra: `best.pt`, `metrics.json` và `split_indices.npz` trong thư mục mới.
Split dự kiến 47.631 train / 4.536 validation / 4.536 KG test triples, với
512 triple mẫu mỗi val/test. Chọn `best.pt` bằng validation filtered MRR.
Chỉ xem KG test sau khi chọn checkpoint; không dùng VG150 test để chọn mô hình.

## 2. C.2 export trên CPU

Sau KG train, Codex hoặc teammate chạy:

```bash
myenv/bin/python -m scripts.export_conceptnet_complex \
  --checkpoint runs/complex/conceptnet-toan-v3-seed42/best.pt \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --output-dir data/knowledge/c2-conceptnet-complex-toan-v1
```

Export kiểm hash KG, review, checkpoint, đúng thứ tự 151 class và 138 valid
bit; từ chối checkpoint TransE/RotatE hoặc output đã tồn tại.

## 3. Ba run PredCls KG-on

Preflight launcher:

```bash
myenv/bin/python -m src.level1_complex_matrix train --max-parallel 3 --dry-run
```

**Toan chạy GPU train rồi GPU evaluate** sau khi C.2 hoàn tất:

```bash
myenv/bin/python -m src.level1_complex_matrix train --max-parallel 3
myenv/bin/python -m src.level1_complex_matrix evaluate --device cuda --max-parallel 1
```

Launcher chạy `kg_on` × ba seed, dùng 57.723 ảnh train và 5.000 ảnh
validation. Có thể chọn một ô bằng `--condition kg_on --seed 42`.
So paired mR@50, R@50 từng seed và mean±sample SD với ConceptNet+TransE,
RotatE, KG-off và GAT thuần. Nếu kết quả cần phân tích cạnh, chạy thêm
`--condition edges_off` × ba seed vì những run đó vẫn dùng vector ComplEx.

Nguồn snapshot ConceptNet và lệnh tái dựng cho teammate có tại
[b2_conceptnet_workflow.md](b2_conceptnet_workflow.md).
