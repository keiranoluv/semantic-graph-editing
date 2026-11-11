# VG150 PredCls — phân tích held-out test của 27 run CLIP

Nguồn: [plan khóa trước test](clip-all-27run-test-plan.json),
[summary test](clip-all-27run-test26446-summary.json), 27 JSON tại `test-v1/`,
[bảng validation](clip-all-27run-val5000-summary.md),
`reports/vg150_graph_statistics.json` và code tại `src/models/gat_kg.py`,
`src/graph/builder.py`, `src/training/sampling.py`.

Toàn bộ 27 report test khớp SHA-256 checkpoint trong plan, cùng hash dữ liệu,
cache CLIP và protocol `PredCls_graph_constrained_top1_per_pair`; mỗi report
đọc đủ 26.446 ảnh, 50 predicate có GT. Ba seed mỗi điều kiện: 42, 123, 3407.
Checkpoint đã được chọn trên validation trước khi mở test. Số dưới đây là
**phần trăm**, `±` là sample standard deviation qua ba seed.

## Kết quả test

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
| --- | ---: | ---: | ---: | ---: |
| MLP + CLIP | 62,35 ± 0,70 | 16,55 ± 1,43 | 65,31 ± 0,23 | 19,36 ± 1,23 |
| GAT + CLIP | 62,64 ± 0,78 | 18,37 ± 0,19 | 65,10 ± 0,76 | 21,04 ± 0,17 |
| GAT + CLIP, KG-off | 62,66 ± 0,40 | 18,25 ± 0,18 | 65,14 ± 0,32 | 20,84 ± 0,37 |
| GAT + CLIP + WordNet TransE | 62,99 ± 0,31 | 17,82 ± 0,12 | 65,66 ± 0,38 | 20,62 ± 0,09 |
| WordNet TransE, edges-off | 62,78 ± 1,38 | 17,95 ± 0,76 | 65,26 ± 1,35 | 20,58 ± 0,58 |
| GAT + CLIP + ConceptNet TransE | 62,90 ± 0,33 | 18,14 ± 0,08 | 65,37 ± 0,34 | 20,77 ± 0,19 |
| ConceptNet TransE, edges-off | **63,41 ± 0,22** | 17,86 ± 0,21 | **65,90 ± 0,20** | 20,60 ± 0,37 |
| GAT + CLIP + ConceptNet RotatE | 63,03 ± 0,42 | **18,39 ± 0,34** | 65,54 ± 0,53 | **21,07 ± 0,18** |
| GAT + CLIP + ConceptNet ComplEx | 63,22 ± 0,20 | 18,14 ± 0,18 | 65,76 ± 0,28 | 20,88 ± 0,17 |

GAT tăng mR@50 **1,82 điểm phần trăm** so với MLP trên test. RotatE nhỉnh
hơn GAT chỉ **0,016 điểm** mR@50 trung bình; paired theo seed là +0,340,
+0,195, −0,487 điểm. Đây là kết quả gần hòa, **không có bằng chứng vững** rằng
RotatE thắng GAT. WordNet TransE, ConceptNet TransE và ComplEx KG-on đều
thấp hơn GAT lần lượt 0,555, 0,232 và 0,228 điểm mR@50. Cả bốn KG-on lại
có R@50 cao hơn GAT 0,25–0,58 điểm: hai metric cho hai góc nhìn khác nhau.

WordNet KG-on thấp hơn **KG-off cùng kiến trúc** 0,436 điểm mR@50 trung bình,
và thấp hơn ở cả ba seed (−0,590/−0,140/−0,580). Đây là ablation trực tiếp
nhất cho thấy vector WordNet hiện tại không cải thiện balanced recall.
KG-off này dùng bundle WordNet; không coi nó là ablation ghép cặp hoàn toàn
cho ba bộ embedding ConceptNet.

## Predicate nào tạo chênh lệch?

Sắp 50 predicate theo số annotation **train**; head = 10 lớp nhiều nhất,
body = 20 lớp tiếp theo, tail = 20 lớp còn lại. Bảng là mean recall@50 trên
test, trước hết tính trung bình ba seed của từng lớp rồi lấy trung bình lớp.

| Điều kiện | Head 10 | Body 20 | Tail 20 |
| --- | ---: | ---: | ---: |
| GAT + CLIP | 50,40 | 16,62 | **4,11** |
| WordNet TransE KG-on | 50,41 | 15,68 | 3,66 |
| ConceptNet TransE KG-on | 50,52 | 16,06 | 4,03 |
| ConceptNet RotatE KG-on | 50,77 | **16,81** | 3,78 |
| ConceptNet ComplEx KG-on | **50,84** | 16,10 | 3,85 |

Không bộ KG-on nào tăng trung bình tail 20 so với GAT; RotatE tăng head/body
nhưng giảm tail. Ví dụ trên test, `on` tăng từ 72,22% recall của GAT lên
73,08% WordNet, 73,61% ConceptNet TransE và 73,69% ComplEx; ngược lại
`riding` giảm từ 45,35% xuống 38,16%/36,65%/35,31%. RotatE đạt 40,60%
`riding` nhưng tăng `walking on` từ 22,13% lên 35,07%. Đây là khác biệt
định lượng theo lớp, chưa chứng minh chính xác cặp nhãn nào bị nhầm lẫn.

Hai ví dụ validation tự động giúp thấy hướng thay đổi, **chưa được người
trong nhóm xác nhận ảnh/box**: ảnh 400, `person–bike`, GT `riding`: GAT dự
đoán `riding` hạng 6, WordNet dự đoán `on` hạng 9; ảnh 16,
`man–street`, GT `walking on`: GAT dự đoán `on` hạng 61, WordNet và RotatE
dự đoán `walking on` hạng 39/37. Không dùng hai ví dụ này để ước lượng tỉ lệ
lỗi.

## Nguyên nhân đã kiểm và giới hạn kết luận

1. **Lệch lớp rất lớn, loss chưa cân bằng lớp.** Train có 405.860 annotation
   quan hệ; head 10 chiếm 88,40%, tail 20 chỉ 1,46%. Các trainer dùng
   cross-entropy không có class weight và lấy tối đa 3 background/positive.
   Mọi mô hình đều có tail recall thấp; KG vector chưa giải quyết được điều đó.
2. **Một phần nhãn quan hệ không vào supervision một nhãn.** Graph builder gán
   `-1` cho cặp object có nhiều predicate khác nhau và sampler bỏ chúng.
   Trên train có 16.894/297.318 positive pairs như vậy (5,68%), chứa
   48.516/405.860 annotation quan hệ (11,95%). Riêng `riding` có 38,21%
   annotation nằm ở các cặp đó; `walking on` 18,99%. Cơ chế này áp dụng cho
   tất cả mô hình, nên giải thích khó khăn chung với một số predicate, **không
   tự nó giải thích riêng việc KG-on kém GAT**.
3. **KG có vào mô hình.** Checkpoint KG-on ghi `use_knowledge=true`, buffer
   có 150/150 vector WordNet hoặc 138/150 vector ConceptNet hợp lệ, norm
   trung bình 1. CPU smoke trên một ảnh validation cho thấy khi zero KG ở
   cùng checkpoint, mean absolute logit delta là 1,52–1,98 tùy embedding.
   Vì vậy không thấy dấu hiệu KG bị tắt nhầm. Đây chỉ là kiểm tra tác động
   lên logits, không đo lợi ích dự đoán.
4. **Độ phủ ConceptNet có giới hạn nhưng không phải lời giải thích duy nhất.**
   Trên test, 93,43% lượt xuất hiện object có vector hợp lệ; 90,71% GT
   relation có cả hai đầu được liên kết. Tuy nhiên `riding` và `parked on`
   lần lượt có 99,19% và 100% GT relation đủ hai đầu liên kết mà vẫn giảm
   recall ở nhiều KG-on. WordNet phủ đủ 150 class vẫn giảm mR@50 so với
   KG-off.
5. **Cách ghép KG hiện tại chỉ là vector khái niệm ở đỉnh.** `KnowledgeGAT`
   nối vector KG đóng băng và valid bit vào đặc trưng object, trong khi GAT
   đã học embedding nhãn object 32D và CLIP vùng ảnh. Không có score KG
   riêng cho bộ ba `(subject, predicate, object)` hoặc cơ chế chọn nguồn
   tri thức theo predicate. Đây là **giả thuyết về nút thắt thiết kế**, phù
   hợp với mẫu head tăng/tail giảm, chưa phải nguyên nhân nhân quả đã chứng
   minh.

Validation có 49 predicate có GT, test có 50: `flying in` vắng ở validation,
chỉ có 5 annotation train và 17 GT triplet duy nhất trên test. Thêm lớp này
làm mR@50 test giảm khoảng 0,33–0,37 điểm tùy mô hình so với khi chỉ tính
49 lớp chung. Tất cả chín điều kiện còn giảm mR@50 từ validation sang test
0,69–2,00 điểm; riêng GAT giảm 1,60, RotatE giảm 0,69. Phần giảm còn lại
có thể liên quan khác biệt phân bố ảnh/quan hệ và chọn checkpoint theo
validation; dữ liệu hiện có không tách được từng nguyên nhân.

## Hướng xử lý sau khi đã xem test

Giữ bảng test này làm kết quả held-out của ma trận đã khóa; không dùng nó để
chọn tiếp mô hình rồi tuyên bố số mới trên cùng test là held-out. Nếu muốn
thử cải tiến, dùng train/validation cho giả thuyết cụ thể: (a) loss cân bằng
predicate và/hoặc xử lý cặp có nhiều nhãn; (b) fusion KG có gating hay score
quan hệ thay vì chỉ nối vector đỉnh. Cần một nguồn đánh giá chưa dùng để
chọn thiết kế, chẳng hạn GQA/PSG ở mức 2, để kiểm tra cải tiến cuối.
