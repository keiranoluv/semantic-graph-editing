# GQA — benchmark bộ nhớ và tốc độ MLP/GAT

Ngày 2026-11-12 (Asia/Bangkok). **MLP và GAT đều chạy được batch 64 graph train
lớn nhất** trên RTX 5090 với cấu hình pilot float32. Đây là benchmark kỹ thuật với
các bước optimizer tạm thời; không lưu checkpoint mới, không dùng điểm của model
tạm để kết luận chất lượng và không mở annotation project test.

**Cấu hình khởi điểm đề xuất: train batch 64, validation batch 32 cho cả MLP/GAT.**
Đây là đề xuất vận hành, chưa khóa thí nghiệm chính hoặc learning rate. Đổi batch
8 → 64 làm đổi số optimizer update/epoch; không coi nó tương đương cấu hình pilot
về học máy, và không tự động nhân learning rate theo batch size.

## 1. Những gì đã đo

- 64 ảnh train có số object **sau chuẩn hóa** lớn nhất, tie-break theo image ID.
  Tất cả là ảnh thật, không nhân bản graph lớn để tạo batch. Cohort được khóa trước
  phép đo; trích CLIP đóng băng từ JPEG trong ZIP vào cache riêng.
- Graph lớn nhất: ảnh `2337434`, 126 object, 15.750 candidate và 798 GT. Đo riêng
  ảnh này và các batch gồm 8/16/32/64 graph lớn nhất ở lượt đầu.
- Train thông thường: đúng 512 ảnh pilot theo thứ tự cohort đã khóa, không chọn
  lại theo metric. Validation: đúng 128 ảnh pilot, giữ cả ảnh không có GT.
- MLP/GAT, embedding object và classifier 311 đầu ra giữ nguyên implementation
  đã chốt; đầu vào là 5D geometry + CLIP 512D + embedding object học được. Seed 42,
  AdamW, gradient clipping 5, deterministic algorithms, float32, không AMP/TF32.
- Train đo đủ CPU collation, lấy mẫu pair như pilot, copy sang GPU, forward,
  soft-target cross entropy, backward, kiểm gradient và optimizer step. GAT luôn
  giữ toàn bộ cạnh message passing, không chỉ những pair được lấy mẫu để train.
- Stress inference chấm **mọi pair**, tính loss hữu hạn, không backward. Phép đo
  này không bao gồm ranking GC/NG; ranking và tổng hợp metric được đo riêng trong
  validation thông thường.
- Validation bao gồm copy logits về CPU, ranking và matcher **cả GC/NG**, tổng hợp
  310 lớp. Các điểm chất lượng của model tạm bị loại khỏi report.

Dùng `perf_counter` và đồng bộ CUDA trước/sau mỗi batch đo, không chỉ đo thời gian
đưa lệnh lên GPU. Warm-up 2 batch; lượt đầu lặp 3 lần, lượt kiểm lại batch 32/64
lặp 5 lần. Có ghi riêng thời gian collation CPU. Hai lượt dùng cùng cohort/cache;
lượt thứ hai chạy sau khi tests đã kết thúc để kiểm lại tốc độ trên máy không có
job test đồng thời. Các model được khởi tạo lại theo từng case, state optimizer
được giữ giữa các bước của cùng case; không dùng trọng số pilot đã train.

## 2. Bộ nhớ trên các graph lớn nhất

GiB dưới đây là **peak CUDA allocated của PyTorch** ở lượt đầu, gồm model, tensor
và AdamW state cho train. Inference không có AdamW state vì chưa gọi optimizer
step. Report cũng lưu peak reserved; cả hai không tính CUDA context hoặc bộ nhớ
của process khác. Máy báo tổng bộ nhớ thiết bị khoảng 31,36 GiB.

| Batch graph lớn | MLP train | MLP inference mọi pair | GAT train | GAT inference mọi pair |
|---:|---:|---:|---:|---:|
| 1 | 0,110 | 0,194 | 0,425 | 0,404 |
| 8 | 0,398 | 0,672 | 1,729 | 1,643 |
| 16 | 0,641 | 1,012 | 2,668 | 2,533 |
| 32 | 0,983 | 1,585 | 4,239 | 4,024 |
| 64 | 1,586 | 2,502 | **6,764** | **6,420** |

Peak reserved cao nhất ở GAT train batch 64 khoảng **7,70 GiB**. Đây là dư địa tốt
cho MLP/GAT hiện tại, không phải bảo đảm cho nhánh KG/fusion, kiến trúc mới, hidden
dimension khác, DDP hoặc process GPU chạy đồng thời. Chưa đo ngưỡng batch tối đa
đến khi OOM; không cần đẩy bộ nhớ sát trần để chọn batch khởi điểm.

Tập 64 ảnh này cực đại theo tổng node/candidate trong một batch 64 ảnh train
khác nhau; không chứng minh cực đại tuyệt đối theo số positive sampled pair hoặc
mọi workspace nội bộ. Do đó vẫn cần giữ kiểm tra tài nguyên khi chạy chính.

## 3. Tốc độ trên subset thông thường

Lượt đầu, ảnh/giây tính từ trung vị thời gian ba lần đo; dữ liệu graph đã có trong
RAM. Validation tính cả GC/NG trên CPU nên không được so như tốc độ GPU thuần.

| Batch | MLP train ảnh/s | GAT train ảnh/s | MLP validation ảnh/s | GAT validation ảnh/s |
|---:|---:|---:|---:|---:|
| 8 | 2.694 | 1.291 | 503 | 461 |
| 16 | 3.413 | 1.929 | 503 | 474 |
| 32 | 3.717 | 2.490 | 498 | 479 |
| 64 | 3.850 | 2.759 | 495 | 476 |

Tăng batch giúp train, đặc biệt GAT, nhưng không cải thiện rõ validation. Batch 64
train vẫn giữ dư địa bộ nhớ trong stress test; batch 32 validation là điểm bắt đầu
hợp lý. Sai khác nhỏ giữa các batch validation không đủ để kết luận mức nào luôn
nhanh nhất.

Collation CPU chiếm phần đáng kể của thời gian: ở batch 64 lượt đầu, khoảng 80%
MLP và 58% GAT. Hàm hiện tại dựng target dense cho mọi pair rồi mới lấy mẫu, đồng
thời tạo lại Python lookup mỗi lượt. Đây là hướng tối ưu launcher sau này, chưa
thay trong benchmark. Không nên cache toàn bộ target dense của full train: riêng
22,44 triệu pair × 311 lớp × 4 byte đã gần 26 GiB; nên cân nhắc biểu diễn sparse
và kiểm tương đương nhãn/loss trước khi thay code.

## 4. Phần chuẩn bị dữ liệu và giới hạn

Lượt đầu mất khoảng 25,04 giây cho chuẩn bị, gồm đọc/chuẩn hóa train, chọn graph,
trích/kiểm cache stress và dựng graph. Trích cache stress mất 6,11 giây; đọc HDF5
và dựng các graph benchmark mất 0,57 giây. Peak RSS host của cả process tại lúc
chuẩn bị khoảng 3,46 GiB; không phải số đo RAM của full dataset đã dựng thành PyG.

Throughput bên trên **không bao gồm** khởi động, đọc toàn bộ annotation, kiểm hash,
trích CLIP, nạp HDF5 từng batch, ghi log/checkpoint, hoặc tranh chấp tài nguyên.
Đây là tốc độ của đường chạy giữ graph trong RAM. Trước khi cam kết thời gian chạy
chính, cần đo ít nhất một epoch end-to-end với cache và loader full train thật.

Cấu hình KG/phương pháp mới phải benchmark riêng sau khi chốt kiến trúc. Chưa lấy
thời gian MLP/GAT để dự báo chi phí nhánh KG hoặc phương pháp 2023+.

## 4a. Kiểm lại và dự toán có điều kiện

Lượt kiểm lại gồm 20 case, mỗi case đo 5 lần sau warm-up; tất cả PASS, không OOM. Dưới đây là tốc độ trung vị của lượt này:

| Model | Train batch 32 (ảnh/s) | Train batch 64 (ảnh/s) | Validation batch 32 GC/NG (ảnh/s) |
|---|---:|---:|---:|
| MLP | 3694 | 3854 | 498 |
| GAT | 2470 | 2720 | 479 |

Ngoại suy tuyến tính từ throughput RAM-resident: `t_epoch = t_train512 × 69942/512 + t_val128 × 5000/128`. Đây là **dự toán**, không phải thời gian full epoch đã chạy; giả định validation chấm cả GC/NG mỗi epoch.

| Model | Train/epoch dự toán (giây) | Validation/epoch dự toán (giây) | Tổng/epoch (giây) | 50 epoch × 3 seed (phút) |
|---|---:|---:|---:|---:|
| MLP | 18.2 | 10.0 | 28.2 | 70.5 |
| GAT | 25.7 | 10.4 | 36.1 | 90.4 |

Hai model chạy tuần tự trên một GPU, kịch bản **50 epoch × 3 seed/model** tương ứng khoảng **2.68 giờ** phần train + validation trong mô hình ngoại suy này. 100 epoch gấp đôi; đây không phải lựa chọn số epoch hoặc lịch train đã khóa. Chưa cộng chuẩn bị full cache/loader, checkpoint, logging, sai khác phân phối graph và phương pháp mới/KG.

Tốc độ train batch 64 tiếp tục tốt hơn batch 32 ở lượt kiểm lại. Dù vậy, lợi ích hiệu năng không chứng minh cấu hình batch 64 đạt cùng chất lượng hay hội tụ sau cùng số epoch. Cần đo một epoch end-to-end thật và kiểm loss/metric khi triển khai launcher chính.

## 5. Artifacts và tái tạo

- [Kết quả lượt đầu: 36 case](gqa-benchmark-v1/results.json).
- [Cohort benchmark](gqa-benchmark-v1/cohort.json): ID ảnh thật, không có test.
- [Lượt kiểm lại batch 32/64](gqa-benchmark-v1-repeat/results.json).
- [Biên bản kiểm chứng](gqa-benchmark-v1/verification.json).

```bash
myenv/bin/python -m tools.benchmark_gqa_scale
myenv/bin/python -m tools.benchmark_gqa_scale \
  --report-dir reports/level2/gqa-benchmark-v1-repeat \
  --batch-sizes 32 64 --repeats 5
```

Các lệnh trên mô tả hai lượt đã chạy; tool từ chối ghi đè `results.json` đã có.
Muốn đo lượt mới dùng `--report-dir` khác. Stress cache mặc định nằm ở
`data/cache/gqa-benchmark-v1/`, chứa feature thật và hash JPEG/record. Khi thay
cohort/encoder/mã liên quan, phải dùng cache mới; không sửa cache pilot. Runtime
khác nhau giữa các lượt nên không yêu cầu hash results giống nhau.

**280 tests pass**, thêm 4 ca kiểm chọn stress cohort, thứ tự instance/feature,
cache sai shape và giá trị không hữu hạn. Test fixture tổng hợp chỉ kiểm code,
không được tính thành ảnh hoặc số liệu benchmark. Không sửa evaluator/report
Level 1, checkpoint/report pilot hoặc notebook cá nhân.
