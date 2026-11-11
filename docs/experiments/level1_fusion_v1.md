# Mức 1: kiểm tra lợi ích KG bằng fusion và đối chứng xáo trộn

> Cập nhật 2026-11-11: đã có [15 run reference-v1 full validation](../../reports/level1/reference-v1-fusion/15run-val5000-analysis.md). Các bảng cũ bên dưới dùng protocol legacy. Sáu run relation prior được để riêng vì bất nhất L1/L2; không dùng cho kết luận mới. Xem [báo cáo gọn](../LEVEL1_REPORT.md).

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

## Kết quả validation (2026-10-26)

Đã train và evaluate đủ **21/21** checkpoint trên toàn bộ 5.000 ảnh val.
Audit CPU xác nhận từng report khớp SHA-256 `best.pt`, epoch và bốn metric
R@50/mR@50/R@100/mR@100 trong checkpoint; 21 report có cùng hash dữ liệu,
CLIP cache, protocol và support theo predicate. Xem [summary JSON](../../reports/level1/clip-fusion-v1-21run-val5000-summary.json).
Đơn vị bảng dưới là **%**, dạng trung bình ± độ lệch chuẩn mẫu của ba seed.

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| `baseline` | 60,864 ± 0,610 | 26,037 ± 0,706 | 63,481 ± 0,470 | 29,051 ± 0,526 |
| `pair_real` | 61,470 ± 0,433 | 26,071 ± 0,215 | 63,849 ± 0,462 | 28,540 ± 0,143 |
| `pair_shuffled` | 61,234 ± 0,111 | 26,228 ± 0,093 | 63,720 ± 0,271 | 28,891 ± 0,289 |
| `visual_real` | 61,681 ± 0,085 | 25,534 ± 0,495 | 64,139 ± 0,091 | 28,193 ± 0,776 |
| `visual_shuffled` | 61,708 ± 0,470 | 26,570 ± 1,093 | 64,273 ± 0,051 | 29,466 ± 0,370 |
| `relation_real` | 61,103 ± 0,766 | 25,904 ± 0,450 | 63,505 ± 0,733 | 28,444 ± 0,681 |
| `relation_shuffled` | 61,191 ± 0,363 | 25,863 ± 0,232 | 63,707 ± 0,424 | 28,462 ± 0,393 |

Paired delta **mR@50** theo thứ tự seed 42/123/3407, đơn vị điểm phần trăm:

| Fusion | Real − baseline | Real − shuffled |
|---|---:|---:|
| Pair | +0,197 / −0,527 / +0,431 (mean +0,034) | −0,251 / +0,082 / −0,303 (mean −0,157) |
| Visual gate | −0,232 / −0,848 / −0,429 (mean −0,503) | −1,451 / +0,646 / −2,303 (mean −1,036) |
| Relation prior | +0,534 / −1,097 / +0,164 (mean −0,133) | +0,343 / −0,187 / −0,032 (mean +0,041) |

Không nhánh KG thật nào vượt **cả** baseline và shuffled trên cả ba seed;
không nhánh nào đạt ngưỡng mean +0,5 điểm đã định trước. `pair_real` tăng
R@50 so baseline +0,606 điểm trung bình (cả ba seed dương), nhưng mR@50 chỉ
+0,034 điểm và thua `pair_shuffled` −0,157 điểm. `visual_real` cũng tăng R@50
+0,817 điểm nhưng giảm mR@50 −0,503 điểm; bản shuffled có mR@50 cao hơn
real +1,036 điểm và dao động lớn giữa seed. `relation_real` gần hòa
`relation_shuffled` về mR@50 (+0,041 điểm), và thấp hơn baseline −0,133 điểm.
Đây là bằng chứng **chưa thấy lợi ích ngữ nghĩa KG ổn định** trong ba kiểu
fusion này; không đủ để khẳng định KG nói chung không hữu ích.

Head/body/tail định nghĩa theo tần suất predicate train (10/20/19 lớp có
support trên val). Recall@50 trung bình theo lớp, đơn vị %:

| Điều kiện | Head 10 | Body 20 | Tail 19 |
|---|---:|---:|---:|
| `baseline` | 45,269 | 31,646 | 10,010 |
| `pair_real` | 46,249 | 31,663 | 9,564 |
| `pair_shuffled` | 45,754 | 30,849 | 11,087 |
| `visual_real` | 44,656 | 31,723 | 8,954 |
| `visual_shuffled` | 45,121 | 30,846 | 12,304 |
| `relation_real` | 45,797 | 31,841 | 9,185 |
| `relation_shuffled` | 45,767 | 31,269 | 9,696 |

Tail chỉ có 444 GT relation trong 5.000 ảnh val, nên các khác biệt nhỏ theo
lớp có độ bất định đáng kể. Val cũng dùng để chọn `best.pt`; đây là so sánh
phương pháp trên tập phát triển, chưa là kiểm định held-out mới. Kết quả test
cũ đã được xem trước khi thiết kế ma trận này.
