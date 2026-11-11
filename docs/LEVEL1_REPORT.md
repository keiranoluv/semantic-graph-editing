# Mức 1 — báo cáo kết quả ngắn

Cập nhật 2026-11-11. Tài liệu để nhóm tổng hợp báo cáo chính; không kèm slide/PDF.

[Hướng dẫn đọc mô hình](LEVEL1_MODEL_GUIDE.md): input/output, ví dụ tính điểm, mục đích 20 cấu hình, checkpoint từng seed và giải thích bất nhất L1/L2 của relation-prior.

## 1. Bài toán và phạm vi

VG150 PredCls: input là ảnh, hộp và nhãn đối tượng có sẵn; output là quan hệ có hướng giữa các cặp đối tượng. So sánh MLP phi đồ thị, GAT và GAT có tri thức nền. CLIP ViT-B/32 đóng băng cung cấp vector vùng ảnh 512D, ghép với hình học 5D và embedding nhãn học được 32D. GAT dùng hai tầng, bốn head, hidden 256; nhánh KG cơ bản ghép thêm vector entity 64D và valid bit trước phép chiếu đỉnh.

Đã có WordNet TransE (mapping 150/150 lớp), ConceptNet TransE/RotatE/ComplEx (138/150 lớp). C.2 export và buffer checkpoint được kiểm tra hash và class ID. Mapping có xác nhận theo lô và giới hạn về nhãn đa nghĩa; không tuyên bố mọi assertion KG đã được người review riêng.

## 2. Dữ liệu và cách chấm

- Train / validation / test: **57.723 / 5.000 / 26.446 ảnh**, chia theo ảnh.
- Bảng chính mục 3–5 là **full validation, reference-v1** (GC); mục 5b bổ sung reference-ng-v1 (NG), 3 seed 42/123/3407, đơn vị %, mean ± sample standard deviation.
- R@K: tỷ lệ GT được tìm lại trong top-K dự đoán, rồi trung bình theo ảnh.
- mR@K: trung bình recall mỗi predicate qua các ảnh chứa predicate đó, rồi trung bình đủ 50 predicate; lớp vắng đóng góp 0.
- Matching theo nhãn object và IoU cả hai hộp ≥0,5; giữ annotation trùng. Top-1 foreground mỗi cặp, xếp hạng bằng full softmax có background.
- **Checkpoint vẫn được chọn bằng metric legacy**; đợt này chỉ chấm lại, không train/chọn lại. Không gọi các bảng này là test độc lập.
- 54 checkpoint được chấm bằng GPU RTX 5090, batch 128. Tần suất/prior chỉ tính từ train. Thống kê train loại 16.894 cặp nhiều predicate, chứa 11,95% annotation; đây là giới hạn supervision.

[Chi tiết evaluator](evaluation/PredCls.md) · [Audit](experiments/level1_audit_2026-11-11.md).

## 3. Ma trận chính — 27 run

| Cấu hình | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| MLP + CLIP | 64.10 ± 0.82 | 17.21 ± 1.16 | 66.61 ± 0.35 | 19.53 ± 1.24 |
| GAT + CLIP | 64.46 ± 0.93 | 19.55 ± 0.14 | 66.52 ± 0.95 | 21.40 ± 0.07 |
| GAT, KG-off | 64.36 ± 0.48 | 19.01 ± 0.21 | 66.38 ± 0.53 | 20.96 ± 0.17 |
| GAT + WordNet TransE | 64.78 ± 0.48 | 18.46 ± 0.65 | 66.98 ± 0.40 | 20.49 ± 0.32 |
| WordNet, edges-off | 65.12 ± 1.13 | 19.47 ± 1.25 | 67.15 ± 1.15 | 21.34 ± 1.16 |
| GAT + ConceptNet TransE | 64.82 ± 0.29 | 19.31 ± 0.59 | 66.87 ± 0.30 | 21.33 ± 0.81 |
| ConceptNet TransE, edges-off | 65.51 ± 0.34 | 19.49 ± 0.94 | 67.59 ± 0.34 | 21.32 ± 0.84 |
| GAT + ConceptNet RotatE | 64.71 ± 0.68 | 18.75 ± 0.53 | 66.79 ± 0.72 | 20.41 ± 0.45 |
| GAT + ConceptNet ComplEx | 64.93 ± 0.35 | 19.05 ± 0.23 | 66.99 ± 0.44 | 21.02 ± 0.28 |

GAT tăng **2,34 điểm mR@50** so với MLP; cả ba seed tăng. Bốn nhánh KG-on đều thấp hơn GAT về mR@50 trung bình, dù R@50 tăng nhẹ. Đây là so sánh kiến trúc, không tách riêng tác động của số tham số. KG-off dùng bundle/kiến trúc WordNet; không coi là đối chứng ghép cặp hoàn toàn cho mọi ConceptNet embedding.

[Nguồn 27 run](../reports/level1/reference-v1/27run-val5000-summary.json).

## 4. KG theo cặp × loss — 12 run

`ce`: cross entropy thông thường; `adjusted`: cộng τ log π vào logits khi tính loss, τ=0,25, prior π chỉ từ train và có add-one smoothing. `off/on`: tắt/bật nhánh KG theo cặp. Inference dùng logits của mô hình.

| Cấu hình | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| ce_off | 65.16 ± 0.10 | 19.18 ± 0.57 | 67.33 ± 0.11 | 20.90 ± 0.42 |
| ce_on | 64.05 ± 0.15 | 19.75 ± 0.99 | 66.17 ± 0.26 | 21.68 ± 1.15 |
| adjusted_off | 61.97 ± 0.12 | 25.80 ± 1.10 | 64.31 ± 0.33 | 27.88 ± 0.94 |
| adjusted_on | 61.97 ± 0.42 | 25.97 ± 0.61 | 64.39 ± 0.66 | 28.48 ± 0.74 |

- Loss điều chỉnh, cùng KG-off: **mR@50 +6,62 điểm**, R@50 **−3,19 điểm**. Paired mR@50: **+6,17 / +7,95 / +5,75**; cải thiện cả ba seed.
- Khi đã điều chỉnh loss, thêm KG: **mR@50 +0,17 điểm**, R@50 gần như không đổi. Paired mR@50: **+0,43 / −1,51 / +1,59**; chưa ổn định.
- Với CE, KG tăng trung bình +0,57 điểm mR@50 nhưng một seed giảm; không đủ kết luận lợi ích ổn định.

[Nguồn và paired delta 12 run](../reports/level1/reference-v1-pair/12run-val5000-analysis.md).

## 5. Fusion và đối chứng KG xáo trộn — 15 run

Cả năm cấu hình dùng loss điều chỉnh và cùng initialization backbone trong mỗi seed. `pair` dùng nhánh MLP cặp KG; `visual` có gate nhìn thêm embedding cặp sau GAT. `real/shuffled` giữ kiến trúc và valid mask, thay mapping embedding giữa các lớp hợp lệ. Một permutation cố định là giới hạn của đối chứng này.

| Cấu hình | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| baseline | 61.96 ± 0.67 | 25.78 ± 0.99 | 64.47 ± 0.50 | 28.42 ± 0.86 |
| pair_real | 62.64 ± 0.41 | 25.72 ± 0.30 | 64.89 ± 0.45 | 27.92 ± 0.29 |
| pair_shuffled | 62.40 ± 0.13 | 25.85 ± 0.09 | 64.77 ± 0.30 | 28.29 ± 0.36 |
| visual_real | 62.81 ± 0.09 | 25.20 ± 0.30 | 65.16 ± 0.06 | 27.68 ± 0.50 |
| visual_shuffled | 62.82 ± 0.49 | 25.97 ± 0.91 | 65.30 ± 0.10 | 28.71 ± 0.05 |

- Pair real so baseline: mR@50 **−0,06 điểm**; so shuffled: **−0,13 điểm**.
- Visual real so baseline: **−0,58 điểm**; so shuffled: **−0,77 điểm**.
- Không nhánh real nào có paired mR@50 dương qua cả ba seed so với cả baseline và shuffled. Chưa chứng minh lợi ích của liên kết ngữ nghĩa KG đúng.
- Mốc +0,5 điểm và cả ba seed dương được đặt trước trên protocol legacy; việc áp dụng lại để đọc bảng reference-v1 là mô tả bổ sung, không phải một kiểm định mới được đăng ký trước.

[Nguồn và paired delta 15 run](../reports/level1/reference-v1-fusion/15run-val5000-analysis.md).

## 5b. Đánh giá bổ sung không giới hạn một predicate mỗi cặp

Đã chấm thêm đủ **54 checkpoint / 18 cấu hình × 3 seed**, GPU RTX 5090, batch 128, cùng 5.000 ảnh validation. `reference-ng-v1` giữ mọi predicate foreground làm ứng viên; top-K vẫn tính trên toàn ảnh. Mỗi bộ ba chiếm một vị trí. Matching/GT/cách trung bình không đổi; không train hay chọn lại checkpoint.

Bảng rút gọn bên dưới có đơn vị %, mean ± sample std. [Bảng đủ 18 cấu hình, K=50/100 và paired delta](../reports/level1/reference-comparison/comparison.md).

| Cấu hình | mR@50 GC | ng-mR@50 | ng-R@50 |
|---|---:|---:|---:|
| MLP + CLIP | 17.21 ± 1.16 | 31.37 ± 2.54 | 76.53 ± 1.92 |
| GAT + CLIP | 19.55 ± 0.14 | 33.21 ± 0.70 | 77.87 ± 0.48 |
| GAT + WordNet | 18.46 ± 0.65 | 31.84 ± 1.25 | 77.69 ± 0.69 |
| Pair: adjusted_off | 25.80 ± 1.10 | 36.06 ± 1.18 | 75.39 ± 0.23 |
| Pair: adjusted_on | 25.97 ± 0.61 | 36.96 ± 1.31 | 75.19 ± 0.28 |
| Fusion: baseline | 25.78 ± 0.99 | 37.13 ± 1.62 | 75.00 ± 0.91 |
| Fusion: pair_real | 25.72 ± 0.30 | 37.93 ± 0.10 | 75.91 ± 0.12 |
| Fusion: pair_shuffled | 25.85 ± 0.09 | 36.62 ± 1.09 | 75.65 ± 0.28 |
| Fusion: visual_real | 25.20 ± 0.30 | 36.98 ± 0.97 | 75.82 ± 0.45 |
| Fusion: visual_shuffled | 25.97 ± 0.91 | 37.51 ± 1.16 | 75.58 ± 0.98 |

- GAT so MLP: ng-mR@50 trung bình **+1,84 điểm**, nhưng một seed giảm (khác kết quả cả ba seed tăng ở GC). Bốn nhánh KG cơ bản đều chưa vượt GAT về ng-mR@50 trung bình; thứ tự có thể khác ở K=100.
- Loss điều chỉnh với KG-off: ng-mR@50 **+3,81 điểm**, cả ba seed tăng; ng-R@50 **−2,62 điểm**. Thêm KG sau loss: **+0,90 điểm ng-mR@50**, paired **+0,60 / −0,33 / +2,41**; chưa ổn định.
- **Pair fusion KG thật có tín hiệu tích cực trong NG:** ng-mR@50 **37,93 ± 0,10%**, hơn shuffled **+1,31 điểm**, cả ba seed tăng (**+0,40 / +1,08 / +2,45**). So baseline, tăng trung bình **+0,80 điểm**, nhưng seed 123 giảm. Chưa chứng minh vượt cả baseline lẫn shuffled ổn định. Đây là quan sát bổ sung, không phải kiểm định ý nghĩa thống kê.
- Visual KG thật vẫn thấp hơn baseline và shuffled về ng-mR@50 trung bình. Không quy kết mọi KG đều vô ích hoặc đều hữu ích từ một chỉ số.
- Điểm NG cao hơn GC là do cách chọn dự đoán để chấm khác nhau. Không dùng ng-mR của mô hình này so với mR GC của mô hình khác.

Ví dụ đã kiểm bằng notebook: ảnh 3, girl → hair có GT `has, has, with, with`. GC chọn `with` hạng 5 và bỏ lỡ `has`; NG có `with` hạng 5 và `has` hạng 7, đều vào top 50. Recall toàn ảnh tăng từ **4/6 lên 6/6**. `miss05` là miss của annotation has; with đã đúng GT.

[Đối chiếu evaluator với upstream](../reports/level1/reference-ng-v1-differential-check.json) · [Kiểm chứng 108 report GC/NG](../reports/level1/reference-comparison/comparison.json) · [Kiểm tra notebook](../reports/level1/reference-comparison/demo-check.json).

## 5c. Long-tail theo tần suất annotation train

Đã tổng hợp **54 checkpoint / 108 report GC–NG** mà không train hoặc inference lại. Nhóm được khóa từ annotation của toàn train: head = 10 lớp nhiều nhất, body = 20 lớp tiếp theo, tail = 20 lớp ít nhất; đồng hạng theo ID tăng dần. Đếm cả annotation trùng và cặp đa predicate, không dùng prior lấy mẫu để chia nhóm.

| Nhóm | Số lớp | Annotation train | Tỷ trọng train | Annotation validation |
|---|---:|---:|---:|---:|
| Head | 10 | 358.780 | 88,40% | 29.468 |
| Body | 20 | 41.145 | 10,14% | 3.238 |
| Tail | 20 | 5.935 | 1,46% | 497 |

Điểm nhóm là trung bình recall theo ảnh của từng predicate trong nhóm; **top-K vẫn tính trên toàn ảnh**, không lọc theo nhóm rồi xếp hạng lại. Lớp `flying in` vắng GT validation vẫn tính 0 trong mẫu số 20 lớp tail. Với mỗi checkpoint/K/protocol, đã xác nhận `mR toàn bộ = 0,2 × head + 0,4 × body + 0,4 × tail`. Không dùng trung bình đều ba nhóm.

Bảng rút gọn cho **nhóm tail**, đơn vị %, mean ± sample std qua ba seed:

| Cấu hình | Tail mR@50 GC | Tail ng-mR@50 | Tail mR@100 GC | Tail ng-mR@100 |
|---|---:|---:|---:|---:|
| MLP + CLIP | 2.45 ± 0.35 | 14.12 ± 2.15 | 3.25 ± 0.67 | 22.79 ± 3.47 |
| GAT + CLIP | 3.87 ± 0.52 | 14.30 ± 0.97 | 4.41 ± 0.84 | 21.57 ± 3.56 |
| Pair: CE, KG-off | 3.65 ± 0.60 | 12.64 ± 2.50 | 4.18 ± 0.56 | 23.70 ± 0.71 |
| Pair: adjusted, KG-off | 9.79 ± 2.33 | 17.19 ± 3.51 | 10.50 ± 2.36 | 28.81 ± 1.02 |
| Pair: adjusted, KG-on | 8.51 ± 0.30 | 18.67 ± 3.08 | 9.73 ± 0.50 | 28.30 ± 0.91 |
| Fusion: baseline | 8.95 ± 0.61 | 20.62 ± 1.37 | 10.52 ± 0.58 | 29.22 ± 1.78 |
| Fusion: KG thật | 8.46 ± 0.65 | 20.21 ± 0.43 | 9.40 ± 0.37 | 29.67 ± 2.36 |
| Fusion: KG xáo trộn | 9.70 ± 1.84 | 18.35 ± 0.82 | 10.98 ± 2.23 | 29.24 ± 1.24 |

**Loss là cải thiện rõ nhất đối với nhóm hiếm trong đối chứng hiện có.** `adjusted_off − ce_off`: tail GC mR@50 **+6,14 điểm**, tail NG ng-mR@50 **+4,56 điểm**; cả ba seed tăng trong cả hai cách chấm. Body cũng tăng, còn head giảm nhẹ. Đây là kết quả của đổi loss với cùng nhánh KG-off, không quy lợi ích này cho KG.

**KG theo cặp phụ thuộc cách chấm.** `adjusted_on − adjusted_off`: tail GC mR@50 **−1,28 điểm**, nhưng tail NG ng-mR@50 **+1,47 điểm**, paired NG **+1,94 / +0,03 / +2,46**. Mức tăng nhỏ ở seed 123 và sự đảo chiều giữa GC/NG chưa đủ để kết luận KG cải thiện mọi kiểu dự đoán quan hệ hiếm; ở NG@100, chênh lệch trung bình lại âm (28,30% so 28,81%).

**Pair fusion KG thật:** tail NG@50 hơn shuffled **+1,86 điểm**, cả ba seed tăng (**+1,25 / +1,93 / +2,41**), nhưng thấp hơn baseline không KG **−0,41 điểm** (20,21% so 20,62%). Tăng ng-mR@50 toàn bộ của nhánh này so baseline đến từ head/body; không được diễn giải thành tăng recall nhóm hiếm ở K=50. Ở K=100, tail NG trung bình cao hơn baseline (29,67% so 29,22%), nhưng chưa phải kết luận thắng ổn định.

**Visual KG thật** thấp hơn baseline và shuffled về tail mR@50 trung bình trong cả GC và NG. Không có bằng chứng chung rằng mọi cách thêm KG đều giúp long-tail.

Giới hạn: tail validation chỉ có 497 annotation; `says` chỉ 1 annotation/1 ảnh, `playing` 3 annotation/2 ảnh, `flying in` không có. Mean/std ba seed phản ánh biến động các run trên cùng validation, không phải khoảng tin cậy về mọi ảnh có thể gặp. Đây là phân tích hậu kiểm; cách chia 10/20/20 là quy ước được công bố của dự án.

[Bảng đủ 18 cấu hình, head/body/tail ở K=50/100, paired delta và danh sách lớp](../reports/level1/long-tail-reference-v1/analysis.md) · [Recall từng lớp và số annotation](../reports/level1/long-tail-reference-v1/per-predicate.csv) · [Định nghĩa nhóm đã khóa](../reports/level1/long-tail-reference-v1/predicate-groups.json).

Chạy lại: `myenv/bin/python -m tools.analyze_level1_long_tail`. Công cụ kiểm hash nguồn/checkpoint, dữ liệu/support và tái dựng mR; từ chối ghi đè nếu kết quả thay đổi. Sáu relation-prior vẫn bị loại. [Biên bản kiểm chứng long-tail](../reports/level1/long-tail-reference-v1/verification.json).

## 5d. Zero-shot: tổ hợp quan hệ chưa gặp trong train

Đã chấm GPU **9 checkpoint fusion**: baseline / pair KG thật / pair KG xáo trộn × ba seed, đủ 5.000 ảnh validation; không train/chọn lại. Vocabulary dùng toàn bộ annotation train trước sampler: **29.283 bộ ba lớp**. Validation có **1.130 annotation zero-shot**, thuộc **851 tổ hợp** trên **722 ảnh**. Mọi lớp thành phần đã gặp riêng lẻ trong train.

Top-K vẫn lấy trên **toàn ảnh**. Chỉ giữ GT tổ hợp chưa gặp làm đích chấm, tính recall từng ảnh rồi trung bình trên 722 ảnh có GT này. Đây là **zR theo ảnh**, khác mR theo predicate. GC chỉ một predicate/cặp có hướng; NG cho phép nhiều predicate/cặp.

Đơn vị %, mean ± sample std ba seed:

| Cấu hình fusion | GC zR@50 | GC zR@100 | NG zR@50 | NG zR@100 |
|---|---:|---:|---:|---:|
| Baseline | 10.95 ± 1.20 | 13.99 ± 0.94 | 17.41 ± 2.25 | 28.06 ± 1.58 |
| KG thật | 11.04 ± 1.28 | 13.78 ± 1.51 | 16.98 ± 1.65 | 28.90 ± 0.89 |
| KG xáo trộn | 10.80 ± 0.70 | 13.30 ± 0.48 | 16.53 ± 0.93 | 27.36 ± 1.11 |

**Tín hiệu tích cực tại NG@100:** KG thật hơn baseline **+0,83 điểm**, hơn shuffled **+1,54 điểm**, cùng dấu ở cả ba seed. Tuy nhiên, seed 3407 chỉ hơn baseline **+0,031 điểm**. Tại NG@50, KG thật lại thấp hơn baseline trung bình **−0,43 điểm**, hai seed giảm. GC cũng không cải thiện ổn định. Kết luận cần giới hạn ở cutoff/protocol cụ thể; ba seed cùng dấu không chứng minh ý nghĩa thống kê.

Đây là phân tích hậu kiểm trên validation và chỉ gồm 9/54 checkpoint hợp lệ. “Chưa gặp” xét annotation train của dự án, không khẳng định vắng trong CLIP tiền huấn luyện/KG ngoài. [Bảng đủ từng seed, cách tính và ví dụ](../reports/level1/zero-shot-reference-v1/analysis.md). Evaluator khớp upstream trên 100 ca tổng hợp + 32 ảnh thật; chín run tái lập metric toàn tập cũ. **239 tests PASS**; [biên bản](../reports/level1/zero-shot-reference-v1/verification.json).

## 6. Sáu run relation prior được để riêng

`relation_real` và `relation_shuffled`, mỗi cấu hình ba seed, **không nằm trong kết luận chính**. TransE được train bằng L1 nhưng artifact relation score dùng L2. Hai loại khoảng cách có thể tạo thứ hạng khác nhau; chưa có bằng chứng sửa sẽ cải thiện kết quả.

Không xóa checkpoint/report cũ và không train lại sáu run này trong lượt hoàn thiện Mức 1. Nếu tiếp tục nghiên cứu, cần tạo artifact L1 version mới và một thực nghiệm riêng. [Danh sách loại và hash](../reports/level1/submission/excluded-relation-prior.json).

## 7. Năm đúng / năm bỏ lỡ

Dùng GAT+CLIP seed 42 đã cố định, 100 ảnh validation đầu được xáo bằng seed 2026. Lấy annotation đầu tiên cho mỗi nhóm trên mỗi ảnh; không chọn theo hình đẹp hoặc mô hình thắng. Hit/miss được kiểm theo reference-v1. Năm ảnh xuất hiện ở cả hai nhóm: đây là minh họa, không phải 10 ảnh độc lập để ước lượng tỷ lệ lỗi.

| Case | Ảnh | GT | Dự đoán cùng cặp | Hạng | Diễn giải sơ bộ |
|---|---:|---|---|---:|---|
| [hit01](../reports/level1/submission/cases/hit01.png) | 40 | chair → near → table | near | 7 | Ghế sát bàn, near phù hợp bố cục. |
| [hit02](../reports/level1/submission/cases/hit02.png) | 7 | book → on → desk | on | 4 | Vật gán nhãn book ở trên bàn; cần xác nhận vật nhỏ. |
| [hit03](../reports/level1/submission/cases/hit03.png) | 89 | flower → in → vase | in | 1 | Hoa cắm trong bình, quan hệ in phù hợp. |
| [hit04](../reports/level1/submission/cases/hit04.png) | 81 | door → on → cabinet | on | 36 | Cánh tủ gắn trên vùng tủ; hộp sát mép ảnh. |
| [hit05](../reports/level1/submission/cases/hit05.png) | 3 | bag → on → table | on | 3 | Túi đặt trên mặt bàn. |
| [miss01](../reports/level1/submission/cases/miss01.png) | 40 | table → near → chair | with | 48 | near bị thay bằng with; cặp đảo hướng của hit01. |
| [miss02](../reports/level1/submission/cases/miss02.png) | 7 | lady → with → hair | has | 12 | with/has có thể gần nghĩa; GT ID không khớp. |
| [miss03](../reports/level1/submission/cases/miss03.png) | 89 | shelf → on → table | on | 64 | Đúng on nhưng hạng 64; nhãn shelf khó xác nhận. |
| [miss04](../reports/level1/submission/cases/miss04.png) | 81 | door → on → cabinet | on | 61 | Đúng on nhưng hạng 61, ngoài top-50. |
| [miss05](../reports/level1/submission/cases/miss05.png) | 3 | girl → has → hair | with | 5 | GT có cả has và with: with đúng; chỉ annotation has bị bỏ lỡ trong GC. |

Đã quan sát ảnh và viết nhận xét AI; **chưa có xác nhận của người trong nhóm**. Người review cần kiểm hộp/nhãn và sửa diễn giải nếu cần. Hai loại lỗi đã thấy: chọn predicate khác (có thể gần nghĩa), và chọn đúng predicate nhưng ranking thấp. Không gán chắc chắn nguyên nhân lỗi cho CLIP/GAT/KG chỉ từ một ảnh.

[Phân tích có ảnh](../reports/level1/submission/cases/analysis.md) · [Phiếu review CSV](../reports/level1/submission/cases/human-review.csv).

## 8. Kết luận sử dụng trong báo cáo

**GAT cải thiện so với MLP trong pipeline của nhóm. Loss điều chỉnh cải thiện mR@50 trên validation và đánh đổi R@50. Các cách bổ sung KG đã thử chưa vượt đồng thời baseline và đối chứng xáo trộn ổn định qua seed. Đánh giá NG bổ sung cho thấy pair fusion KG thật vượt shuffled ở cả ba seed, nhưng chưa vượt baseline ở cả ba seed.**

Không kết luận KG nói chung vô ích; không so số legacy với reference-v1; không coi mR@50≈26% là accuracy hay bằng chứng ngang SOTA. Tương thích evaluator không đồng nghĩa preprocessing/training đã giống mọi paper. Test cũ đã được xem trước khi thiết kế các nhánh mới, nên không phải holdout độc lập cho các cải tiến sau đó.

Zero-shot bổ sung cho thấy pair fusion KG thật hơn hai đối chứng ở NG@100 qua ba seed; chưa cải thiện nhất quán ở NG@50/GC. Đây là kết luận riêng trên 9 checkpoint, không thay cho kết luận metric toàn tập ở trên.

## 9. Chạy lại và demo

```bash
# Từ thư mục gốc repository; giữ các checkpoint/dữ liệu/cache hiện có.
myenv/bin/python -m tools.level1_reference_matrix --matrix pair --device cuda --batch-size 128
myenv/bin/python -m tools.level1_reference_matrix --matrix fusion --device cuda --batch-size 128
myenv/bin/python -m pytest -q
myenv/bin/jupyter lab notebooks/level1_reference_demo.ipynb
```

Launcher bỏ qua report đã có chỉ khi hash/protocol đúng; không ghi đè. Sau khi thay code evaluator, dùng `--output-dir` mới nếu chấm lại GC để tránh từ chối source hash lịch sử. Cách chạy NG cho cả ba ma trận ở [protocol](evaluation/PredCls.md). Để tạo lại case, chạy `myenv/bin/python -m tools.prepare_reference_cases` ở checkout/output chưa có `cases.json`; công cụ từ chối ghi đè bộ case hiện hữu.

[Notebook demo mới](../notebooks/level1_reference_demo.ipynb) chạy CPU trên một ảnh, có hộp, dự đoán lại và R@50/R@100. Mặc định miss03 minh họa đúng predicate nhưng ngoài top-50; đổi CASE_ID thành hit01 để xem case đúng. Kịch bản 3 phút: input → forward → ranking → matching → giới hạn một ảnh. Demo không thực hiện biến đổi đồ thị/chỉnh sửa ảnh của Mức 2–3.

Môi trường và dataset/checkpoint cần thiết: [hướng dẫn thiết lập](TEAM_SETUP_LEVEL1.md), [tái lập](TEAM_REPRODUCIBILITY_AND_PROTOCOLS.md). Notebook đã chạy tại workspace hiện tại; chưa có bằng chứng chạy lại trên máy teammate. Nhóm cần thử máy trình bày, chuẩn bị video ≤5 phút và xác nhận với giảng viên phạm vi demo Mức 1 nếu cần.

## 10. Bằng chứng và việc nhóm cần xác nhận

Đã hoàn thành 54 lượt evaluation reference-v1 full validation: 27 main + 12 pair/loss + 15 fusion. Hash checkpoint, report, evaluator và support theo lớp được kiểm tra. Dữ liệu máy đọc: [results.json](../reports/level1/submission/results.json). Không train thêm trong đợt này. Hiện **239 tests PASS**, gồm kiểm tra long-tail và 6 ca mới cho zero-shot. Đã chấm thêm 54 lượt NG và đối chiếu 108 report GC/NG: checkpoint/dữ liệu/support khớp, mean/sample std được tính lại. Cả hai mode khớp evaluator tham chiếu trên 100 ca tổng hợp + 16 ảnh thật. [Kiểm tra bổ sung](../reports/level1/reference-comparison/verification.json); [biên bản GC trước đó](../reports/level1/submission/verification.json).

Còn việc của nhóm: xác nhận 10 case; tổng hợp báo cáo chính, phân công và khai báo đóng góp/AI; thử demo trên máy dùng vấn đáp; xác nhận yêu cầu demo với thầy. Chưa đánh dấu các việc này hoàn thành thay nhóm. Thông tin thành viên chưa được yêu cầu ở lượt hiện tại.

## Tài liệu tham chiếu

- Đề `Project01_Graph_Editing.pdf`, trang 3–6; [bản trong repo](../Project01_Graph_Editing.pdf).
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903): nguồn ý tưởng attention trên đồ thị; mô hình dự án là triển khai GAT đơn giản.
- [CLIP — Radford et al., 2021](https://proceedings.mlr.press/v139/radford21a): nguồn encoder thị giác tiền huấn luyện, dùng đóng băng trong dự án.
- [TransE — Bordes et al., 2013](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html): nguồn mô hình embedding KG.
- [Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314): nguồn ý tưởng điều chỉnh logits; prior và τ của dự án được mô tả ở trên.
- [Evaluator tham chiếu, commit cố định](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/ceb71fa88461c2a97a6258a80f47669d89207296/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py).

Ảnh minh họa lấy từ VG150 đã tải, giữ image ID/URL/hash trong cases.json; không tuyên bố nhóm sở hữu ảnh. Tri thức WordNet/ConceptNet giữ provenance và attribution theo các manifest hiện có. AI đã hỗ trợ kiểm tra code, chạy evaluation, tổng hợp số liệu và soạn nhận xét ảnh; người trong nhóm chịu trách nhiệm kiểm chứng và giải thích khi vấn đáp.
