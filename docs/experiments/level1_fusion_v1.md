# Mức 1: kiểm tra lợi ích KG bằng fusion và đối chứng xáo trộn

## Câu hỏi và giới hạn

Ma trận trước trên VG150 PredCls + CLIP cho `adjusted_on − adjusted_off` chỉ
+0,27 điểm phần trăm mR@50 trung bình ba seed; một seed giảm 1,20 điểm.
Nhánh KG đã hoạt động nhưng gate dựa trên class có mean 0,97–0,995 trên mọi
cặp class hợp lệ. Thử nghiệm này kiểm tra riêng (a) fusion nào có lợi ích dự
đoán so với GAT và (b) việc **liên kết đúng** lớp VG với ConceptNet có ích hơn
vector bị xáo trộn hay không. Không có kết quả dương nào được giả định trước.

Thiết kế này được đặt ra **sau khi nhóm đã xem VG150 test của 27 run cũ**.
Mọi kết quả mới trên chính test ấy chỉ có ý nghĩa thăm dò, không phải một
held-out độc lập. Các run dưới đây dùng train để học và 5.000 ảnh val để chọn
checkpoint/so sánh phương pháp. Không dùng nhãn VG test khi tạo score KG,
huấn luyện hay chọn checkpoint.

## Ma trận đã khóa trước train GPU

Giữ CLIP ViT-B/32 vùng GT box, nhãn object, hình học, graph đầy đủ không phụ
thuộc nhãn predicate, GAT 2 tầng, ConceptNet TransE C.2 reviewed 138/150
class, train 57.723/val 5.000 ảnh, 3 seed 42/123/3407, AdamW, batch 512,
negative ratio 3, tối đa 200 epoch, patience 15. **Cả bảy điều kiện dùng
cùng logit-adjusted CE với `tau=0.25` và cùng prior chỉ lấy từ train.**
Mọi model khởi tạo cùng backbone GAT và tất cả các nhánh fusion theo cùng thứ
tự; condition chỉ chọn nhánh nào tham gia forward. Trong cùng seed, trọng số
GAT ban đầu giống nhau. `baseline` trả chính xác logits GAT, nhờ đó là đối
chứng GAT được train lại trong cùng phiên bản code và protocol.

| Điều kiện | Nhánh cộng vào logits GAT | KG |
|---|---|---|
| `baseline` | Không | Không dùng |
| `pair_real` | Pair MLP + gate chỉ nhìn KG, như nhánh trước | Thật |
| `pair_shuffled` | Cùng pair MLP/gate | Xáo trộn |
| `visual_real` | Pair MLP + gate nhìn embedding cặp sau GAT và KG | Thật |
| `visual_shuffled` | Cùng visual gate | Xáo trộn |
| `relation_real` | MLP trên 10 score TransE quan hệ ConceptNet của cặp lớp | Thật |
| `relation_shuffled` | Cùng relation prior | Xáo trộn |

Tổng cộng **7 × 3 = 21 run**. `visual_*` dùng gate riêng cho từng predicate,
nhận `[h_s,h_o,z_s,z_o,z_s*z_o,|z_s-z_o|,valid_s,valid_o]`. Correction có
biên độ tối đa 2 logit qua `2*tanh` để tránh nhánh KG áp đảo logits GAT như
gate class-only cũ. `relation_*` dùng 10 điểm
`-||e_s + r_ConceptNet − e_o||₂`, chuẩn hóa theo từng loại cạnh trên các cặp
class VG có liên kết. Chúng đi qua MLP để thành 51 correction logits (gồm
background), có sigmoid scale riêng theo predicate và cùng biên độ ±2.
Hai nhánh mới là **hai phương pháp riêng**, không kết hợp trong 21 run này.

Shuffled control dùng một derangement cố định `seed=20261025` chỉ trên 138
class có vector. Không class hợp lệ nào giữ vector gốc; 13 class không liên kết
và valid mask được giữ nguyên. Ma trận score KG được hoán vị đồng thời ở hai
trục class, nên phân bố score và số tham số không đổi. Một permutation là
đối chứng sàng lọc; nếu KG thật thắng, cần lặp lại với nhiều permutation và
kiểm trên holdout mới trước khi nói lợi ích ngữ nghĩa có tính tổng quát.

## Nguồn score và chạy trên máy teammate

`data/knowledge/c2-conceptnet-relation-scores-toan-v1/` là artifact CPU cục
bộ, không commit. Lệnh tạo lại từ bundle reviewed và C.1 `best.pt`:

```bash
myenv/bin/python -m scripts.prepare_pair_relation_scores
```

Script kiểm SHA-256 checkpoint C.1 với manifest C.2, hash quan hệ/class IDs,
chỉ đọc embedding KG và không đọc nhãn VG predicate. Teammate tải dữ liệu/ảnh
và tái tạo C.1/C.2 theo [TEAM_SETUP_LEVEL1.md](../TEAM_SETUP_LEVEL1.md) và
[conceptnet_review_and_runs.md](../knowledge/conceptnet_review_and_runs.md).

Từ thư mục gốc repository, sau khi commit code/tài liệu:

```bash
myenv/bin/python -m src.level1_fusion_matrix train --max-parallel 6 --dry-run
myenv/bin/python -m src.level1_fusion_matrix train --max-parallel 6
```

Launcher duy trì tối đa sáu process và mở run kế tiếp khi slot trống; không
ghi đè run/log đã có. Muốn chạy một ô: thêm `--condition visual_real --seed 42`.
Run ở `runs/level1-clip-fusion-v1/`, log ở `logs/level1-clip-fusion-v1/`.

Chỉ sau khi **21/21 train xong** và launcher in `exit=0`, chạy evaluator GPU:

```bash
myenv/bin/python -m src.level1_fusion_matrix evaluate --device cuda --max-parallel 1
```

Lệnh này tạo 21 report val và
`reports/level1/clip-fusion-v1-21run-val5000-summary.json`.
Đọc mR@50 là metric chính; báo cáo cả R@50, R@100, mR@100, từng seed,
head/body/tail và độ lệch chuẩn. So **trong cùng mode**: real − shuffled,
real − baseline, shuffled − baseline. Mốc bằng chứng mạnh hơn cần real KG có
paired mR@50 dương ở cả ba seed so với **cả baseline và shuffled**, mean gain
ít nhất 0,5 điểm phần trăm; đây là ngưỡng thực dụng đặt trước run, không phải
kiểm định thống kê với chỉ ba seed. Báo cáo mọi tradeoff R@50 và không chọn
condition theo một seed hoặc một lớp có lợi.
