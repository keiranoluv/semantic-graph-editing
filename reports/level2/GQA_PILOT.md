# GQA pilot v1 — kết quả thật của MLP/GAT với CLIP đóng băng

**Đã hoàn thành lượt kiểm pipeline:** 512 ảnh train, 128 ảnh validation, seed 42,
5 epoch cho mỗi model. Cả hai có loss hữu hạn, học được quan hệ và chấm lại được
checkpoint tốt nhất. **Chưa hoàn thành thực nghiệm chính GQA hoặc Level 2.**
Không dùng project test để chọn subset, train, chọn checkpoint hay điều chỉnh tham số.

## 1. Input và output

Input của mỗi ảnh gồm box/nhãn object GT, 5 đặc trưng hình học chuẩn hóa và vector
CLIP ViT-B/32 512 chiều cho từng crop thật. CLIP dùng trọng số OpenAI đã có tại máy,
`eval()` và `requires_grad=False`; không huấn luyện detector/encoder/sinh ảnh.
Embedding nhãn object là phần được học: 1.700 dòng, gồm 1.699 lớp và ID 0 để dành.
Mỗi model trả logits **311 chiều** cho mỗi cặp object có hướng: background +
310 predicate. Không dùng bảng KG hoặc checkpoint model đã train trên VG150.
Tái sử dụng trọng số CLIP gốc không phải pretrain trên annotation VG150.

MLP phân lớp từ hai vector object; GAT truyền thông tin qua tất cả cặp có hướng
trước khi phân lớp. Topology không dùng quan hệ GT. GAT thêm self-loop bên trong
GATConv; tập cặp cần dự đoán vẫn không có self-pair. Attributes không được đưa vào
đặc trưng. Đây là **PredCls dự đoán quan hệ**, chưa phải model nhận câu yêu cầu rồi
sinh thao tác chỉnh sửa đồ thị.

## 2. Subset và provenance

[Cohort đã khóa](gqa-pilot-v1-cohort.json) được ghi trước khi trích feature và chạy
model. Với từng split train/val trong manifest GQA v1, sắp theo
`SHA256(gqa-pilot-v1:42:split:image_id)` rồi lấy 512/128 ảnh đầu tiên. Không chọn lại
ảnh theo chất lượng annotation, số node hoặc metric. Thứ tự hash độc lập với thứ tự
JSON; validation vẫn tách theo ảnh khỏi train. Vocabulary giữ nguyên từ toàn bộ
train dự án, không thu hẹp theo pilot hay mở rộng theo validation.

| Thống kê | Train | Validation |
|---|---:|---:|
| Ảnh trong cohort | 512 | 128 |
| Object hợp lệ | 8.350 | 2.124 |
| Candidate pair | 164.902 | 40.418 |
| Positive pair | 26.222 | 6.065 |
| Pair có nhiều predicate khác nhau | 377 | 102 |
| GT annotation sau chuẩn hóa | 26.651 | 6.175 |
| Predicate có GT trong subset | 120/310 | 80/310 |
| Ảnh không có GT quan hệ | 16 | 7 |
| Ảnh không có candidate (dưới 2 node) | 6 | 0 |
| Self-relation đã loại | 21 | 12 |
| Box được clip | 108 | 19 |

Subset này không có dòng GT trùng hệt bộ ba; code vẫn giữ đúng quy ước cho trường
hợp trùng và đã được kiểm bằng test tổng hợp. Test tổng hợp chỉ kiểm code, không
được tính thành dữ liệu hoặc kết quả model thật. Không có OOV object/predicate bị
loại trong cohort này; audit GQA đầy đủ vẫn có các trường hợp OOV ở held-out split.

Cache đọc thẳng JPEG trong ZIP, giữ thứ tự instance của adapter. Có **2 ảnh train**
khác kích thước JPEG so với annotation; box crop được scale theo từng chiều trước
khi floor/ceil. Geometry vẫn theo hệ tọa độ annotation. Mỗi ảnh lưu SHA-256 của
JPEG gốc và record đã chuẩn hóa trong HDF5. Manifest cache khóa hash cohort,
manifest nguồn, trọng số CLIP, mã trích feature và file feature hoàn tất.

- [Metadata cache và số liệu xử lý](gqa-pilot-v1/cache.json).
- [Config + hash nguồn chạy thực tế](gqa-pilot-v1/config.json).
- Cache lớn ở `data/cache/gqa-pilot-clip-v1/`, checkpoint ở
  `checkpoints/level2/gqa-pilot-v1/{mlp,gat}/best.pt`; các binary không nằm trong Git.
- HEAD nền là `6a15317`; mã mới chưa commit khi chạy nên hash từng file là provenance
  chính xác của implementation được dùng. Hash commit lịch sử trong tài liệu Level 1
  được giữ nguyên.

## 3. Train và lựa chọn checkpoint

Launcher GQA riêng; không gọi launcher VG150 có kiểm cứng 51 đầu ra. Dùng lại class
MLP/GAT hiện có với kích thước truyền tường minh: feature 517, embedding 32,
hidden 256; GAT hai layer, bốn head, dropout 0,1. AdamW, learning rate 0,001,
weight decay 0,01, batch 8 ảnh, clip gradient norm 5; bật deterministic algorithms.
Mỗi epoch shuffle ảnh. RNG shuffle/sampling được reset cùng seed cho hai model,
nên chúng dùng cùng thứ tự ảnh và cùng cặp giám sát.

Một pair có hai predicate đúng nhận target 0,5/0,5; pair có một predicate nhận
one-hot; pair không được gán quan hệ nhận background. Dòng GT trùng không tăng
trọng số train. **Loss là soft-target cross entropy, không phải BCE đa nhãn độc lập.**
Cách này giữ giám sát cho tất cả positive pair thay vì bỏ pair đa predicate như
projection single-label cũ; softmax vẫn buộc các predicate cạnh tranh xác suất.

Mỗi ảnh train dùng toàn bộ positive pair và tối đa 3 background/positive. Ảnh
không có GT dùng tối đa 32 background; ảnh không có candidate không đóng góp loss
nhưng vẫn nằm trong cohort và thống kê. Nhãn background dựa trên quan hệ chưa được
annotation; không chứng minh rằng mọi quan hệ đó sai trong ảnh thật.

Validation loss dùng **tất cả** candidate pair và cùng soft target. Vì train lấy
mẫu background, không so trực tiếp trị số train loss với validation loss như cùng
một phân phối. Chọn checkpoint bằng **GC mR@50 trên validation**, epoch sớm hơn
nếu bằng điểm; không dùng NG hoặc test để chọn. Cả hai chọn epoch 5; chưa có early
stopping/convergence, chưa tìm siêu tham số, chưa có nhiều seed.

## 4. Protocol chấm

Wrapper GQA gọi matcher reference PredCls đã khóa của Level 1 **không sửa file
matcher**, truyền tường minh `num_predicates=310` khi tổng hợp và kiểm đầu ra 311
lớp trước khi chấm. Tên protocol trong JSON ghi rõ GQA custom/GC hoặc NG/310 lớp.
Không gắn nhãn GQA200/GQA100 hoặc so trực tiếp với điểm paper có taxonomy khác.

- Softmax gồm cả background, sau đó chỉ xếp hạng foreground; object score bằng 1
  vì dùng nhãn/box GT. Ranking diễn ra riêng cho từng ảnh.
- GC: tối đa một predicate foreground cho mỗi pair. NG: xếp hạng mọi pair ×
  predicate foreground, cho phép nhiều predicate cùng pair. Giữ chính sách tie
  của matcher reference hiện có, không đổi để làm đẹp kết quả.
- Khớp bằng nhãn subject/predicate/object và IoU box subject/object ≥ 0,5, dùng
  quy ước pixel inclusive XYXY của reference. Vì thế có thể khớp instance khác
  cùng nhãn với box đủ trùng; không phải exact instance-ID matching.
- Giữ mọi dòng GT, kể cả dòng trùng. Một dự đoán có thể hit nhiều dòng GT tương ứng.
- **R@K:** tỷ lệ GT được tìm thấy trong top K của từng ảnh, rồi lấy trung bình
  trên ảnh có GT. Validation có 121/128 ảnh đóng góp recall; 7 ảnh còn lại vẫn
  được forward, tính loss và báo số lượng, không gán recall 0 hoặc 1 tùy tiện.
- **mR@K:** với mỗi predicate, lấy trung bình recall trên những ảnh có GT của
  predicate đó; sau đó trung bình trên **đủ 310 predicate**, lớp không có support
  nhận 0. Không âm thầm đổi mẫu số thành 80 lớp thấy trong validation pilot.

## 5. Kết quả

Tất cả số trong bảng là **%**, cao hơn tốt hơn. Đây là kết quả checkpoint đã reload,
không phải số minh họa. [MLP JSON](gqa-pilot-v1/mlp.json), [GAT JSON](gqa-pilot-v1/gat.json)
chứa lịch sử từng epoch, support và recall từng lớp, hash checkpoint.

| Model | GC R@50 | GC R@100 | GC mR@50 | GC mR@100 | NG R@50 | NG R@100 | NG mR@50 | NG mR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 30,988 | 47,135 | 0,340 | 0,501 | 31,349 | 48,238 | 0,349 | 0,579 |
| GAT | 29,866 | 46,780 | 0,315 | 0,463 | 30,817 | 48,795 | 0,331 | 0,558 |

MLP train loss giảm 2,0753 → 0,7820; GAT giảm 1,1223 → 0,7669. GC R@50 trước
train chỉ 0,778% với MLP và 0% với GAT. Loss và gradient hữu hạn xuyên suốt pilot.
Cải thiện này xác nhận model nhận được tín hiệu học, chưa chứng minh khả năng
khái quát tốt hoặc ưu thế của GAT.

Validation có 2.752 dòng `to the left of` và 2.752 dòng `to the right of`, tổng
**89,13%** trong 6.175 GT. Cả hai model mới hit bốn loại predicate ở GC@50:
trái, phải, `on`, `wearing`; nhiều loại hiếm chưa có hit. Ngoài thiên lệch này,
230/310 lớp vắng trong validation cũng góp phần làm mR nhỏ. Ngay cả khi chỉ đổi
mẫu số sang 80 lớp có support thì mR@50 vẫn chỉ khoảng 1,32%/1,22%; đây chỉ là
phép giải thích mẫu số, **không thay metric chính**. Cần cohort lớn hơn để đánh giá
long-tail; chưa kết luận MLP tốt hơn GAT từ một pilot.

## 6. Chi phí quan sát

RTX 5090, torch 2.7.0+cu128. Thời gian do mã chạy đo tại máy, không phải ước lượng
cho full dataset; cache filesystem và kích thước graph có thể ảnh hưởng mạnh.

| Bước | Thời gian đo | Thông tin khác |
|---|---:|---|
| CLIP 640 ảnh / 10.474 crop | 19,01 giây | Cache 11.965.312 byte (~11,41 MiB) |
| MLP, 5 epoch + initial/final validation | 2,03 giây | 415.671 tham số; peak CUDA allocated ~144,50 MiB |
| GAT, 5 epoch + initial/final validation | 2,98 giây | 540.087 tham số; peak CUDA allocated ~263,35 MiB |

Thời gian model tính từ sau khi dataset đã load vào RAM và model/optimizer đã tạo;
bao gồm train/eval/lưu best checkpoint, không bao gồm toàn bộ khởi động, đọc/kiểm
hash annotation/cache và ghi report cuối. Peak là PyTorch allocated, không phải
VRAM tổng của `nvidia-smi`. GAT có chi phí candidate/message edge N(N−1), cần kiểm
đuôi phân phối số object trước khi nâng batch hoặc chạy toàn bộ.

## 7. Chạy lại và demo

Từ root repo, dùng đúng Python môi trường dự án:

```bash
myenv/bin/python -m tools.cache_gqa_pilot
myenv/bin/python -m tools.train_gqa_pilot
myenv/bin/python -m tools.demo_gqa_pilot
myenv/bin/python -m pytest -q
```

Cache hoàn tất được xác minh hash và dùng lại khi cùng identity. Launcher train
cố ý từ chối ghi đè run/report đã tồn tại; muốn chạy lại cùng cấu hình:

```bash
myenv/bin/python -m tools.train_gqa_pilot \
  --output-dir checkpoints/level2/gqa-pilot-v1-rerun \
  --report-dir reports/level2/gqa-pilot-v1-rerun
```

Demo mặc định dùng GAT và **ảnh đầu tiên theo cohort validation đã khóa**, ID
`2407303` (7 object, 42 pair, 11 GT), không tìm ảnh đẹp theo metric. In top 10 quan
hệ thật dự đoán từ checkpoint, score và cờ khớp annotation theo instance ID.
[Output demo đã lưu](gqa-pilot-v1/demo-gat.json) có cả dự đoán khớp và không khớp.
Cờ khớp tự động không phải human review, cũng không kết luận quan hệ vắng annotation
nhất định sai về thị giác. Chọn `--image-index N` hoặc đổi `--report` sang MLP;
`--output PATH` ghi JSON bất biến theo nội dung. Đây là CLI đơn giản, không cần UI
web hoặc thay notebook Level 1 của người dùng.

## 8. Kiểm chứng và phần còn thiếu

**273 tests pass**, gồm 13 ca mới: deterministic cohort, chống thay manifest,
nhãn 310/311, GT trùng/đa predicate, background và graph 0/1 node, GC/NG,
batch tương đương single-image, GT relation không thay input dự đoán, gradient
MLP/GAT hữu hạn. [Biên bản kiểm chứng](gqa-pilot-v1/verification.json) ghi hash và
các kiểm tra artifact bổ sung. Mọi hook pre-commit áp dụng đã pass.

Chưa train nhánh KG GQA, chưa chọn/tái lập phương pháp 2023+, chưa có 300 cặp
MagicBrush, chưa có human review (vẫn 0), chưa train model thao tác. GQA dùng ảnh
Visual Genome, nên hai thí nghiệm riêng không chứng minh chuyển miền độc lập.
Pilot đủ để chuyển sang thiết kế thực nghiệm chính: khóa quy mô/split/evaluator,
audit graph lớn và phủ lớp, chạy ba seed cùng input, xây mapping KG GQA riêng.
Không mở rộng tự động hoặc ghi đè các kết quả Level 1 trong lượt pilot này.
