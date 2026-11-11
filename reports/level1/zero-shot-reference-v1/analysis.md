# Zero-shot PredCls — chín checkpoint fusion

Cập nhật 2026-11-11. Chạy GPU trên đủ 5.000 ảnh validation cho ba cấu hình × ba seed 42/123/3407. Giữ checkpoint đã chọn trước bằng legacy mR@50; không train hoặc chọn lại checkpoint. Sáu relation-prior bất nhất L1/L2 tiếp tục bị loại.

## Câu hỏi và cách chấm

**Model có dự đoán được tổ hợp quan hệ chưa từng xuất hiện trong annotation train không?** Input/output vẫn là PredCls: ảnh qua đặc trưng CLIP đóng băng, GT box/nhãn object và các đặc trưng của từng nhánh → 51 logits cho mỗi cặp object có hướng (50 predicate + background). Zero-shot chỉ thay nhóm GT được chấm, không đổi input/model.

Bộ ba dùng **lớp** `(subject, predicate, object)`, không dùng node ID. Tập đã gặp lấy từ toàn bộ **405.860 annotation của 57.723 ảnh train**, kể cả dòng trùng và các cặp đa predicate bị sampler bỏ qua. Có **29.283 bộ ba khác nhau**. Bộ ba không thuộc tập này là zero-shot; điều đó không có nghĩa object/predicate mới hay quan hệ vắng trong ConceptNet.

Validation có **1.130 annotation zero-shot / 33.203 annotation tổng cộng (3,40%)**, thuộc **851 bộ ba khác nhau**, trong **722/5.000 ảnh**. Tất cả thành phần object/predicate đã từng xuất hiện riêng lẻ trong train, ở vai trò tương ứng. Đây là kiểm tra tổ hợp mới.

Cách tính với mỗi K = 50 hoặc 100:

1. Xếp hạng dự đoán **toàn ảnh** theo GC hoặc NG. GC chỉ giữ một predicate/cặp có hướng; NG cho phép nhiều predicate/cặp. Dự đoán bộ ba đã gặp vẫn chiếm chỗ trong top-K.
2. Giữ các dòng GT zero-shot làm đích chấm. Matching như reference-v1: đúng predicate, lớp subject/object và IoU của cả hai hộp ≥0,5; giữ dòng annotation trùng.
3. Tính recall từng ảnh = số dòng GT zero-shot được khớp / tổng dòng GT zero-shot của ảnh.
4. Lấy trung bình đều trên **722 ảnh có GT zero-shot**. 4.278 ảnh còn lại không đóng góp vào trung bình này. Sau đó lấy mean ± sample std (ddof=1) qua ba seed.

`zR@K = mean_i (matched_zero_shot_GT_i@K / zero_shot_GT_i)`, với các ảnh có mẫu số >0.

**Ví dụ minh họa:** một ảnh có hai GT zero-shot, một quan hệ được khớp ở hạng 45 và quan hệ còn lại ở hạng 70. Ảnh đó đạt zR@50 = 1/2, zR@100 = 2/2. Các quan hệ đã gặp trong ảnh không vào mẫu số, nhưng dự đoán của chúng vẫn ảnh hưởng thứ hạng. Nếu hai ảnh lần lượt đúng 1/1 và 1/3 thì điểm trung bình là (100% + 33,33%)/2 = 66,67%, không phải pooled recall 2/4 = 50%.

**zR khác mR:** zR ở đây trung bình theo ảnh trên GT tổ hợp mới; mR trung bình theo lớp predicate. Long-tail xem predicate hiếm, còn zero-shot xem tổ hợp chưa gặp; một predicate phổ biến cũng có thể tạo tổ hợp zero-shot.

## Kết quả

Đơn vị %, mean ± sample std của ba seed trên cùng validation:

| Cấu hình | GC zR@50 | GC zR@100 | NG zR@50 | NG zR@100 |
|---|---:|---:|---:|---:|
| Fusion baseline (không KG) | 10.95 ± 1.20 | 13.99 ± 0.94 | 17.41 ± 2.25 | 28.06 ± 1.58 |
| Pair fusion KG thật | 11.04 ± 1.28 | 13.78 ± 1.51 | 16.98 ± 1.64 | 28.90 ± 0.89 |
| Pair fusion KG xáo trộn | 10.80 ± 0.70 | 13.30 ± 0.48 | 16.53 ± 0.93 | 27.36 ± 1.11 |

### Từng seed

| Cấu hình | Seed | GC zR@50 | GC zR@100 | NG zR@50 | NG zR@100 |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 9.642 | 12.943 | 14.838 | 26.609 |
| baseline | 123 | 11.223 | 14.270 | 18.379 | 27.834 |
| baseline | 3407 | 11.996 | 14.755 | 19.012 | 29.748 |
| pair_real | 42 | 12.043 | 14.847 | 18.880 | 27.991 |
| pair_real | 123 | 11.486 | 14.429 | 16.087 | 28.917 |
| pair_real | 3407 | 9.593 | 12.056 | 15.979 | 29.779 |
| pair_shuffled | 42 | 11.507 | 13.751 | 17.020 | 27.129 |
| pair_shuffled | 123 | 10.786 | 13.351 | 17.112 | 28.562 |
| pair_shuffled | 3407 | 10.102 | 12.793 | 15.454 | 26.379 |

### Chênh lệch ghép cặp cùng seed

Đơn vị **điểm phần trăm**, thứ tự seed 42 / 123 / 3407:

| Đối chiếu | Metric | Trung bình ± sample std | Từng seed |
|---|---|---:|---|
| KG thật − baseline | zR@50 | +0.087 ± 2.407 | +2.401 / +0.263 / -2.404 |
| KG thật − baseline | zR@100 | -0.212 ± 2.324 | +1.904 / +0.159 / -2.699 |
| KG thật − baseline | ng-zR@50 | -0.428 ± 3.889 | +4.042 / -2.292 / -3.033 |
| KG thật − baseline | ng-zR@100 | +0.832 ± 0.710 | +1.383 / +1.083 / +0.031 |
| KG thật − pair_shuffled | zR@50 | +0.242 ± 0.656 | +0.536 / +0.700 / -0.509 |
| KG thật − pair_shuffled | zR@100 | +0.479 ± 1.053 | +1.096 / +1.078 / -0.737 |
| KG thật − pair_shuffled | ng-zR@50 | +0.453 ± 1.444 | +1.860 / -1.026 / +0.525 |
| KG thật − pair_shuffled | ng-zR@100 | +1.539 ± 1.631 | +0.862 / +0.355 / +3.400 |

## Diễn giải

**Có tín hiệu tích cực tại NG@100:** KG thật đạt **28,90%**, so với **28,06%** baseline và **27,36%** shuffled. KG thật cao hơn cả hai đối chứng ở cả ba seed; mức tăng trung bình tương ứng **+0,83 / +1,54 điểm phần trăm**. Tuy nhiên, so baseline ở seed 3407 chỉ tăng khoảng **+0,031 điểm**. Ba seed cùng dấu không phải chứng minh ý nghĩa thống kê.

**Không cải thiện chung ở mọi cutoff:** tại NG@50, KG thật **16,98%**, thấp hơn baseline **17,41%** trung bình (−0,43 điểm; hai seed giảm). So shuffled **16,53%**, trung bình tăng +0,45 điểm nhưng một seed giảm. Trong GC, chênh lệch so baseline rất nhỏ ở K=50 và âm ở K=100; dấu thay đổi theo seed.

Kết luận dùng trong báo cáo: *Trong chín checkpoint fusion đã cố định, KG thật cho tín hiệu cải thiện recall các tổ hợp chưa gặp ở NG@100 so với cả baseline và shuffled. Hiệu quả phụ thuộc cutoff và protocol; chưa có bằng chứng KG cải thiện zero-shot một cách nhất quán ở NG@50 hoặc GC.* Không chọn riêng metric thắng làm đại diện toàn bộ thực nghiệm.

## Kiểm chứng và giới hạn

- **239 tests PASS**, gồm 6 ca kiểm zero-shot: train-only, lớp/hướng của bộ ba, giữ ranking toàn ảnh, cặp đa predicate và GT trùng, image-macro, ảnh không có zero-shot.
- Đối chiếu trực tiếp `SGZeroShotRecall` và `SGNGZeroShotRecall` ở commit tham chiếu cố định: **100 ca tổng hợp + 32 ảnh thật**, khớp support, hits@50/100 và recall. Số metric trong differential-check là của ca kiểm trộn, không phải benchmark model.
- Cả chín lượt GPU đều tái lập R/mR toàn validation của GC/NG và per-predicate như report lịch sử (sai số metric tối đa được phép 1e-12). Kiểm độc lập bằng NumPy từ hit/miss từng ảnh: **36 giá trị zR** và mean/std/paired delta đều khớp.
- Tập vocabulary/cohort/checkpoint, dữ liệu và mã evaluator có SHA-256. Tool từ chối ghi đè khác nội dung; có thể tiếp tục từ report đã xác minh.
- Đây là phân tích hậu kiểm trên validation, không phải test độc lập; chỉ khảo sát **9/54 checkpoint hợp lệ**. Mean/std qua seed phản ánh biến động các run trên cùng ảnh, không phải khoảng tin cậy tổng quát. Đối chứng KG xáo trộn dùng một permutation cố định.
- “Chưa gặp” chỉ xét annotation của split train trong dự án; không chứng minh CLIP tiền huấn luyện hoặc KG ngoài chưa từng chứa thông tin đó. Annotation thiếu vẫn có thể ảnh hưởng định nghĩa. Không dùng số này để tuyên bố ngang SOTA khi chưa đồng nhất split/preprocessing/protocol.

## Chạy lại và dữ liệu

```bash
myenv/bin/python -m tools.level1_zero_shot --prepare-only
myenv/bin/python -m tools.level1_zero_shot --device cuda --batch-size 128
myenv/bin/python -m pytest -q
```

Output mặc định là thư mục này. Nếu thay evaluator/định nghĩa, dùng `--output-dir` mới; không thay file đã khóa.

[Tổng hợp JSON](9run-val5000-summary.json) · [Vocabulary train](train-seen-triplets.json) · [Cohort validation](validation-cohort.json) · [Checkpoint cố định](frozen-checkpoints.json) · [Differential check](differential-check.json) · [Biên bản kiểm chứng](verification.json).

Mỗi `*-seed*-val5000.json` lưu support/hits@50/100 cho từng ảnh có zero-shot GT, đủ để tính lại metric. [Protocol](../../../docs/evaluation/PredCls.md) · [Hướng dẫn model](../../../docs/LEVEL1_MODEL_GUIDE.md).
