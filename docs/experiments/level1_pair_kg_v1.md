# VG150 PredCls: KG theo cặp và logit adjustment

## Thiết kế

Thử nghiệm 2 × 2 trên cùng VG150 train/val, CLIP ViT-B/32 vùng ảnh đã cache,
GAT hai tầng và ConceptNet TransE C.2 đã review của Toan
(`data/knowledge/c2-conceptnet-toan-v3`). Mỗi điều kiện dùng seed 42, 123, 3407:

| Điều kiện | Nhánh KG theo cặp | Loss train |
|---|---|---|
| `ce_off` | Tắt; logits đúng bằng GAT gốc | Cross entropy |
| `ce_on` | Bật | Cross entropy |
| `adjusted_off` | Tắt | Logit-adjusted cross entropy |
| `adjusted_on` | Bật | Logit-adjusted cross entropy |

Tổng cộng **12 run**. Bốn điều kiện dùng cùng đặc trưng ảnh, split, graph,
batch 512, negative ratio 3, AdamW, tối đa 200 epoch, patience 15 và chọn
checkpoint theo validation mR@50. Nhánh KG không nhận nhãn predicate GT:

```text
box_s, box_o, object_label_s, object_label_o, CLIP_s, CLIP_o
          │
          ├── GAT (hình học + CLIP + embedding nhãn học được) ── logits_base[51]
          │
          └── ConceptNet vector của hai lớp ── pair MLP ── gate ── Δlogits[51]

logits = logits_base + gate × Δlogits
```

Nhánh cặp dùng `[z_s, z_o, z_s*z_o, |z_s-z_o|, valid_s, valid_o]`.
`gate = sigmoid(Linear(pair)) × valid_s × valid_o`; vì vậy nếu một lớp không
được liên kết với ConceptNet thì correction bằng 0. `kg_off` bỏ hẳn phép cộng
và trả chính xác logits GAT. Gate khởi tạo nhỏ (`bias=-3`) để nhánh KG không
làm lệch lớn điểm dự đoán lúc bắt đầu train. Vector KG được đóng băng; pair
MLP/gate và GAT được học từ train PredCls.

Loss điều chỉnh là `CE(logits + 0.25 × log(p_train), target)` trong **train**.
`p_train` được tính từ 57.723 ảnh train theo đúng quy tắc tất cả positive đơn
nhãn và tối đa 3 background mỗi positive, cộng 1 vào mọi lớp trước khi chuẩn
hóa. Khi validation/inference dùng logits gốc. Công thức dựa trên
[logit adjustment cho long-tail learning](https://research.google/pubs/long-tail-learning-via-logit-adjustment/);
`tau=0.25` được cố định trước 12 run, không chọn bằng VG150 test.

## Chuẩn bị và chạy

Chạy từ thư mục gốc repository. Prior đã được tạo tại
`reports/level1/clip-pair-kg-train-prior-v1.json`; để tạo lại trên máy teammate:

```bash
myenv/bin/python -m scripts.prepare_pair_kg_prior
```

Nguồn dữ liệu, link tải JPEG/annotation, lệnh dựng CLIP cache và kiểm tra hash
cho teammate nằm ở [TEAM_SETUP_LEVEL1.md](../TEAM_SETUP_LEVEL1.md).
Teammate cần export cùng C.2 ConceptNet reviewed; launcher kiểm tra 138/150
lớp được liên kết.

```bash
myenv/bin/python -m src.level1_pair_kg_matrix train --max-parallel 6 --dry-run
myenv/bin/python -m src.level1_pair_kg_matrix train --max-parallel 6
```

Launcher lấp đầy tối đa 6 slot: một run hoàn thành sẽ mở run tiếp theo. Có thể
chạy riêng một ô bằng `--condition ce_on --seed 42`; các condition là `ce_off`,
`ce_on`, `adjusted_off`, `adjusted_on`. Output mới ở
`runs/level1-clip-pair-kg-v1/`, log ở `logs/level1-clip-pair-kg-v1/`.
Launcher dừng nếu thấy log/output đã tồn tại; không ghi đè checkpoint cũ.

Sau khi 12 run train xong, người dùng chạy đánh giá GPU trên **toàn bộ val**:

```bash
myenv/bin/python -m src.level1_pair_kg_matrix evaluate --device cuda --max-parallel 1
```

Lệnh này tạo 12 report và summary mean/std 3 seed ở
`reports/level1/clip-pair-kg-v1-12run-val5000-summary.json`.
Đọc hiệu ứng KG bằng `ce_on − ce_off` và `adjusted_on − adjusted_off`,
hiệu ứng loss bằng `adjusted_off − ce_off`. Xem cả R@50 và mR@50 từng seed;
không khẳng định KG thắng baseline trước khi có số liệu.
Nếu KG-on tốt hơn, chênh lệch vẫn gồm cả tác động tăng số tham số của pair
MLP; cần thêm control vector KG xáo trộn hoặc vector ngẫu nhiên cùng kích thước
nếu muốn quy lợi ích riêng cho **ngữ nghĩa** ConceptNet.

VG150 test đã được xem trong thí nghiệm trước; vì vậy kết quả test bổ sung
sau khi thiết kế nhánh này chỉ mang tính thăm dò, không còn là phép kiểm định
held-out độc lập cho lựa chọn kiến trúc. Ưu tiên chốt trên val và sau đó dùng
dataset/holdout mới khi cần xác nhận tổng quát hóa.
