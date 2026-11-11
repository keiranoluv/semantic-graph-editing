# GQA — audit độ phủ và quy mô thực nghiệm chính

Ngày 2026-11-12 (Asia/Bangkok). Pilot đã được khóa bằng commit **`f36ff65`**.
Audit này đọc annotation của **69.942 ảnh train và 5.000 ảnh validation dự án**;
không mở annotation official val (project test), không chạy model hoặc chọn lại
ảnh theo kết quả validation. [Số liệu đầy đủ](gqa-scale-audit-v1.json) bao gồm
support từng predicate và hash nguồn/mã tái tạo.

**Đề xuất dùng toàn bộ 69.942 train / 5.000 validation cho thực nghiệm chính.**
Nếu phải giảm ngân sách, 30.000 train là phương án thử nghiệm trung gian, không
phải bộ dữ liệu đã được khóa trong lượt audit này. Cần đo bộ nhớ/tốc độ trên graph
lớn bằng launcher dự kiến trước khi triển khai huấn luyện chính.

## 1. Phương pháp audit

Dùng đúng manifest GQA v1 và hàm chuẩn hóa đã kiểm chứng: clip box, loại box/OOV
không hợp lệ cùng quan hệ liên quan, loại self-relation, giữ GT đa predicate và
dòng trùng. Thống kê sau chuẩn hóa mới phản ánh dữ liệu model thực sự nhận.
Không thu hẹp vocabulary 310 predicate, không gom đồng nghĩa/số nhiều.

Các cohort train 512, 5.000, 10.000, 20.000, 30.000, 50.000 và toàn bộ là **các
prefix lồng nhau** của thứ tự `SHA256(gqa-pilot-v1:42:train:image_id)`. Prefix 512
khớp chính xác pilot đã khóa; nhãn không tham gia chọn thành viên. Report lưu hash
list ID có thứ tự và mã tái tạo. Validation chính luôn là 5.000 ảnh đã tách từ đầu;
128 ảnh pilot chỉ được thống kê riêng để đối chiếu.

Đếm riêng annotation rows, bộ ba duy nhất, pair có quan hệ và số ảnh có support.
Một predicate xuất hiện 20 lần trên một ảnh không tương đương 20 ảnh có predicate
đó. Percentile dùng NumPy linear interpolation; candidate pair là N(N−1).

## 2. Độ phủ predicate theo quy mô train

“Có ≥20 GT” là ít nhất 20 dòng quan hệ sau chuẩn hóa; “Có ≥20 ảnh” là xuất hiện
trong ít nhất 20 ảnh khác nhau. Các cột đều đếm số lớp trong vocabulary 310.

| Train ảnh | Object/crop | GT rows | Có ≥1 GT | Có ≥5 GT | Có ≥20 GT | Có ≥20 ảnh |
|---:|---:|---:|---:|---:|---:|---:|
| 512 (pilot) | 8.350 | 26.651 | 120 | 43 | 21 | 19 |
| 5.000 | 82.896 | 257.001 | 262 | 157 | 66 | 56 |
| 10.000 | 165.156 | 514.059 | 296 | 212 | 99 | 90 |
| 20.000 | 331.133 | 1.031.943 | 305 | 264 | 143 | 128 |
| 30.000 | 494.323 | 1.530.920 | 308 | 291 | 186 | 162 |
| 50.000 | 821.489 | 2.534.981 | 310 | 302 | 233 | 210 |
| **69.942 (toàn bộ)** | **1.148.828** | **3.541.725** | **310** | **304** | **268** | **248** |

30.000 ảnh vẫn thiếu `higher than` và `jumping in`; toàn bộ train chỉ có lần lượt
2 và 3 GT của hai lớp này. Chuyển từ 30.000 sang toàn bộ train tăng số lớp đạt
20 GT từ 186 lên 268, và đạt 20 ảnh từ 162 lên 248. Do đó chỉ nhìn “308/310 lớp
đã xuất hiện” sẽ đánh giá quá cao độ đủ của cohort nhỏ.

Ngay cả toàn bộ train vẫn có **42 lớp dưới 20 GT**, trong đó **6 lớp dưới 5 GT**:
`bigger than`, `higher than`, `jumping in`, `longer than`, `playing at`, `shorter than`.
Không có predicate nào mất hoàn toàn sau chuẩn hóa. Vẫn cần báo cáo long-tail và
biến thiên giữa seed; tăng dữ liệu không biến các lớp cực hiếm thành lớp đủ mẫu.

## 3. Mất cân bằng là đặc tính toàn bộ dữ liệu

| Predicate | Train đầy đủ | Validation đầy đủ |
|---|---:|---:|
| to the left of | 1.578.919 | 111.947 |
| to the right of | 1.578.919 | 111.947 |
| on | 84.299 | 5.832 |
| wearing | 50.128 | 3.492 |
| of | 31.593 | 2.373 |
| near | 30.856 | 2.009 |
| in | 29.085 | 2.102 |
| behind | 17.700 | 1.277 |
| in front of | 17.688 | 1.257 |
| holding | 9.614 | 700 |

Trái/phải chiếm **89,16% train và 89,20% validation đầy đủ**, gần mức 89,13% của
validation pilot. Vì vậy mR thấp trong pilot không chỉ do chọn ít ảnh: cơ chế train
hiện tại nhận rất nhiều giám sát trái/phải. Tăng quy mô giúp phủ lớp hiếm nhưng
không tự cân bằng objective. Đây là bằng chứng để thiết kế ablation sau này;
audit này chưa thay loss, sampler hoặc protocol đã khóa.

Train đầy đủ có 55.528 pair đa predicate; validation có 3.874. Không phát hiện
dòng GT trùng hệt bộ ba trên hai split sau chuẩn hóa. Vẫn giữ hỗ trợ GT trùng trong
pipeline để không thay quy ước evaluator.

## 4. Validation và cách đọc metric

Validation chính có **267/310 predicate**, 251.006 GT, 4.847 ảnh có GT quan hệ.
43 lớp không có GT validation vẫn nằm trong mẫu số mR=310 theo protocol pilot.
Với quy ước đó, ngay cả khi recall mọi lớp có support đều bằng 1, mR validation
tối đa cũng chỉ **267/310 ≈ 86,13%**. Đây là tính chất mẫu số, không phải lỗi model;
phải báo support khi diễn giải mR. Không đổi validation hoặc taxonomy để nâng trần.

Mọi predicate có GT trong validation đầy đủ đều có GT ở train đầy đủ. Train
30.000 ảnh thiếu một predicate có mặt ở validation; train 20.000 thiếu ba. Số này
chỉ là chẩn đoán độ phủ, không dùng nhãn validation để thêm ảnh train theo lớp.
Không xem hoặc điều chỉnh theo phân bố test trong audit này.

## 5. Kích thước graph và chi phí cấu trúc

| Chỉ số | Train đầy đủ | Validation đầy đủ |
|---|---:|---:|
| Object trung bình/ảnh | 16,43 | 16,42 |
| Object p50 / p90 / p95 / p99 | 16 / 27 / 31 / 41 | 16 / 27 / 31 / 41 |
| Object tối đa | 126 | 69 |
| Candidate tối đa/ảnh | 15.750 | 4.692 |
| Tổng candidate | 22.440.076 | 1.601.542 |
| Ảnh có ≥32 object | 3.205 | 237 |
| Ảnh có ≥64 object | 31 | 2 |
| Ảnh có ≥100 object | 3 | 0 |
| Ảnh không có GT quan hệ | 2.208 | 153 |
| Ảnh không có candidate | 957 | 61 |

Graph train lớn nhất là ảnh **`2337434`**, 126 object, 15.750 candidate, 798 GT;
hai ảnh tiếp theo là `2409218` (112 object) và `497948` (104 object). Report JSON
liệt kê 10 graph lớn nhất mỗi cohort để tái tạo stress test, không chọn theo metric.

Riêng CLIP float16 cho toàn bộ train+validation cần **1.260.491.776 byte ≈ 1,174 GiB**,
chưa gồm metadata HDF5. Đây là phép đếm chính xác kích thước payload vector, không
phải dự báo kích thước file cuối. Tensor logits float32 `[15750,311]` của graph
train lớn nhất chiếm **18,69 MiB**; không bao gồm target, GAT edge activation,
gradient, optimizer, workspace hoặc nhiều ảnh cùng batch. Không suy ra VRAM peak
hay bảo đảm batch vừa GPU chỉ từ số logits này.

Theo sampler pilot, toàn bộ train có 13.607.862 pair giám sát/epoch trước khi chia
batch. GAT vẫn truyền thông tin qua đủ 22.440.076 cạnh có hướng mỗi epoch, ngoài
self-loop nội bộ. Chưa ngoại suy thời gian train từ pilot vì chi phí đọc cache,
collation và graph đuôi lớn chưa được benchmark ở quy mô này.

## 6. Khuyến nghị thực hiện sau audit

1. **Ưu tiên toàn bộ 69.942 train và giữ 5.000 validation.** Dữ liệu nhiều hơn có
   lợi rõ cho support các lớp hiếm; dung lượng vector riêng không phải nút thắt
   lớn. Nếu cần vòng thử phương pháp nhanh, dùng prefix 30.000 như một vòng phát
   triển riêng, công bố đủ hai lớp vắng và không trộn kết luận với full train.
2. Trước chạy chính, đo forward/backward trên graph lớn nhất và một batch nhiều
   graph lớn; kiểm batch theo tổng số candidate/edge nếu cần. Không âm thầm cắt
   object hoặc bỏ ảnh lớn để tránh OOM.
3. Giữ 310 predicate, ảnh rỗng, giám sát đa predicate và protocol GC/NG như đã
   công bố. Những thay đổi loss/sampling phải được khóa và so sánh như ablation;
   báo thêm support/nhóm tần suất dựa trên **train**, không dùng test để tạo prior.
4. Sau khi chốt phương pháp 2023+, mới khóa manifest thí nghiệm chính, cache lớn,
   mapping KG và lịch ba seed. Đây là bước tiếp theo; audit hiện tại không tự
   triển khai các phần đó.

## 7. Tái tạo và kiểm chứng

```bash
myenv/bin/python -m tools.audit_gqa_scale
myenv/bin/python -m pytest -q
```

Tool từ chối ghi đè report khác nội dung; thay nguồn hoặc logic thì dùng đường dẫn
`--report` mới. Chạy lại cùng nguồn giữ nguyên hash output. Test kiểm riêng cách
đếm GT trùng/đa predicate, cohort hash lồng nhau, graph rỗng và chặn mở annotation
test. Tổng **276 tests pass** (thêm 3 test audit); pre-commit pass. Tổng annotation
sau chuẩn hóa khớp report chuẩn bị GQA trước đó; cohort 512/128 khớp số pilot.
[Biên bản kiểm chứng](gqa-scale-audit-v1-verification.json).

Notebook cá nhân giữ nguyên hash khi chốt pilot và audit. Không push remote, train
model mới, tạo cặp MagicBrush hoặc đánh dấu human review trong bước 1–2 này.
