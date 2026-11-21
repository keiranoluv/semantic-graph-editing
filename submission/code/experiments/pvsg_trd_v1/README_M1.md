# PVSG TRD — M1: parser, decoder, evaluator và graph batching CPU

M1 triển khai protocol `pvsg-gt-trd-tiouv1` đã chốt ở [M0](README.md), trong cùng vùng code riêng. **Không có KG, model training, GPU workload hoặc download mới.** Giữ nguyên tất cả artifacts đã khóa ở M0, gồm README lịch sử, schemas, protocol và manifest 338/62. Tài liệu này mô tả phần bổ sung M1.

**Kết quả:** 97 tests CPU đạt (18 M0 + 79 M1); demo giả đạt R/mR 100% bằng logits dựng từ nhãn giả để kiểm tra luồng xử lý. Đây không phải accuracy của mô hình hoặc kết quả benchmark PVSG. Evaluator được đối chiếu với một implementation độc lập bằng Python sets, không gọi lại hàm interval/matching của implementation chính.

**Dữ liệu chưa sẵn sàng train:** audit cấu trúc annotation local phát hiện 34/338 video train cần xử lý ở M2. M1 hoàn thành phần code và kiểm tra semantics; không chứng nhận masks/video hoặc annotation sạch.

## Các module và API

| Module | API / chức năng |
|---|---|
| [contracts.py](contracts.py) | Schema M0 + kiểm tra semantic: unique track IDs, thing/stuff, frame bounds, identity, finite score; union duplicates, visibility filtering |
| [intervals.py](intervals.py) | Inclusive interval ↔ binary timeline, union overlap/adjacency, true temporal set IoU, cửa sổ phủ frame cuối |
| [data.py](data.py) | `AnnotationIndex`, `parse_video`, `parse_targets`, `visibility_from_masks`, `rasterize_targets`; input và supervision có API riêng |
| [graphs.py](graphs.py) | `build_graph`, `batch_graphs`: spatial/temporal/self edges, query edges độc lập, offsets và sample membership |
| [decoding.py](decoding.py) | `WindowLogits`, `merge_window_logits`, `decode_windows`: logits theo frame gốc → hợp cửa sổ → nhiều predicate/intervals |
| [evaluation.py](evaluation.py) | R/mR@20/50/100, tIoU 0,3/0,5/0,7, support/hits theo lớp và nguồn, predicate-only và mean tIoU@100 |
| [cli.py](cli.py) | `demo`, `evaluate`, `audit-annotation`; tất cả chạy CPU, đọc file local, xuất JSON cùng hashes input/code/protocol |
| [synthetic.py](synthetic.py) | Video giả 10 frame, ba track IDs không liên tiếp, occlusion gap, multi-label và negative pairs |

`AnnotationIndex(path)` kiểm tra annotation byte hash, split IDs và vocabulary qua validator M0 trước khi sử dụng. `ids("train")` trả 338 IDs; `ids("test")` trả 62 IDs thuộc `val` upstream. `ids("fit")`/`ids("validation")` báo lỗi khi internal-val chưa khóa, không tự dùng 62 test video để tuning.

`parse_video(record, source=..., visibility=..., mask_reference=...)` chỉ đọc metadata/object table, không đọc relations, captions, QA hoặc summary. Visibility bắt buộc có đủ track IDs, được lấy từ masks; không suy ra từ GT relation spans hoặc mặc định mọi track hiện diện cả video. `parse_targets(record, video)` là bước riêng cho supervision/evaluation.

`visibility_from_masks` nhận iterable các mask 2D integer đã decode, đúng `T,H,W` của annotation. Nó kiểm tra số frame và ID lạ. Void IDs phải được chỉ định tường minh, ví dụ `ignored_mask_ids=(0,)` chỉ khi ID 0 thực sự là void và không trùng object. Đọc PNG/RGB-packed masks, kiểm tra overlay và alignment media là công việc M2; chưa tự đoán cách encode mask.

## Semantics đã thực thi

- Timeline 0-based, intervals `[start,end]` inclusive; cửa sổ trong API là `[start,stop)` của Python. Không đổi FPS, trừ offset, cắt interval vượt biên hoặc ánh xạ predicate lạ tự động.
- Target `[P,L,57]` multi-label với loss mask `[P,L]` theo visibility. Pair order toàn video, gồm mọi ordered pair có đồng hiện; negative pairs vẫn có loss mask hợp lệ. Cắt cửa sổ giữ đúng lát timeline, không tạo supervision boundary giả.
- GT trùng `(s,p,o)` được union trước visibility filtering. GT hết support sau filtering làm audit/evaluation thất bại. Prediction trùng được union và lấy max score; prediction không còn frame hợp lệ được bỏ theo quy tắc decoder. Các số đếm prediction/false positive trong report là sau bước chuẩn hóa này.
- Decoder yêu cầu coverage cho mọi frame đồng hiện của mọi candidate, kể cả negative pairs. Thiếu cửa sổ cuối hoặc bỏ một pair phải báo lỗi. Padding phải bỏ trước khi tạo `WindowLogits`; chỉ số frame luôn là frame gốc của video.
- Ghép bằng mean **logits** trước sigmoid, threshold `>= 0.5`, không gộp gap hoặc bỏ đoạn ngắn. Cộng các cửa sổ theo thứ tự canonical để kết quả không phụ thuộc thứ tự chúng tới. Score là mean probability trên frame active hợp lệ. Chưa mở tùy chọn tuning decoder; thay đổi cần run config có version ở milestone sau.
- Rank một lần trên toàn video sau union; tie-break `(-score, subject_id, predicate_id, object_id)`. Không softmax/background, không graph constraint, không cắt prediction bằng GT relation span.
- R gộp tổng hits/tổng GT; mR trung bình recall của các lớp có GT support. Lớp không support có recall `null` và bị loại khỏi mẫu số mR. Split không có GT xuất metrics `null`; negative videos vẫn được giữ và báo false-positive diagnostics. Mean tIoU@100 tính trên mọi GT, gán 0 cho tuple bỏ sót hoặc nằm ngoài top 100. Mọi recall và mean tIoU trong report dùng **thang phần trăm**; `temporal_iou()` riêng trả `[0,1]`.
- API evaluator yêu cầu input/target/prediction có cùng tập video IDs; phải xuất document `predictions: []` cho video không có dự đoán. Không âm thầm bỏ video khó/thiếu output. Với benchmark chính thức, truyền `expected_video_ids` hoặc CLI `--manifest ... --split ...` để kiểm tra đủ cohort. Grid K/threshold khác mặc định chỉ phục vụ diagnostics và được đánh dấu `official_metric_grid=false`.

Ví dụ GT active 2/10 frame, prediction active cả 10 frame, cả hai track luôn hiện diện: tIoU = 0,2, không hit ở 0,5. Điều này khác logic full `calculate_viou` đã khảo sát trong OpenPVSG. M1 không triển khai nhánh compatibility hoặc tái lập benchmark predicted-track CVPR 2023.

Graph node metadata là `[original_frame, track_id, class_id]`; **không đưa cả ba cột trực tiếp vào model làm features**, đặc biệt track ID chỉ là khóa. Spatial edges là ordered pairs trong frame; temporal edges chỉ nối cùng track ở frame liền kề, hai chiều, không bắc qua occlusion gap. `temporal_delta_frames` giữ chiều thời gian khi batch. Self-loops được xuất riêng; layer tương lai phải tránh thêm lần hai. Bỏ spatial/temporal message-passing edges không xóa query edges. Hai cửa sổ cùng video vẫn là hai graph samples độc lập khi batch.

## Chạy lại trên CPU

Chạy từ repository root, Python 3.12.3 với dependencies trực tiếp trong [requirements-cpu-m1.txt](requirements-cpu-m1.txt). Không cần PyTorch, PyG, MMDetection hay CUDA cho M1; chưa khóa môi trường train/features.

```bash
python experiments/pvsg_trd_v1/validate_m0.py --require-selected
python -m pytest -q experiments/pvsg_trd_v1/tests
python -m experiments.pvsg_trd_v1.cli demo --output /tmp/pvsg_m1_demo.json
```

Ví dụ evaluation với fixtures M0: có hai GT tuples và chỉ dự đoán đúng một tuple, nên R/mR@20 tại 0,5 đều bằng 50%:

```bash
python -m experiments.pvsg_trd_v1.cli evaluate \
  --videos experiments/pvsg_trd_v1/examples/synthetic_video.json \
  --targets experiments/pvsg_trd_v1/examples/synthetic_targets.json \
  --predictions experiments/pvsg_trd_v1/examples/synthetic_predictions.json \
  --output /tmp/pvsg_m1_example_metrics.json
```

Các input trên nhận một document hoặc một list documents theo schemas M0. Kết quả không truyền manifest được ghi `custom_cohort`; không gọi đó là điểm trên official test. Khi đánh giá dữ liệu thật sau này, thêm `--manifest experiments/pvsg_trd_v1/manifests/pvsg_author_release_338_62.lock.json --split test`; fixture nhỏ sẽ bị từ chối vì không đủ 62 test IDs.

Nếu annotation M0 còn ở local, kiểm tra cấu trúc của **train**:

```bash
python -m experiments.pvsg_trd_v1.cli audit-annotation \
  --annotation data/external/pvsg_m0/pvsg.hf_7e5f1ec.json \
  --output /tmp/pvsg_m1_train_annotation_audit.json
```

Lệnh cuối hiện trả **exit code 2** vì phát hiện lỗi dữ liệu; vẫn ghi report để điều tra. Đây là kết quả audit dự kiến của artifact đang khóa, không phải một lần kiểm tra “pass”. Không cần annotation thật để chạy tests/demo.

## Kết quả audit và bàn giao M2

[Report cấu trúc train](provenance/m1_train_annotation_audit.json) dùng đúng annotation hash của M0, chỉ báo **lỗi đầu tiên của từng video**:

| Loại lỗi đầu tiên | Số video train |
|---|---:|
| Predicate ngoài vocabulary 57 lớp | 17 |
| Interval vượt timeline metadata | 16 |
| Self-pair hoặc track tham chiếu không hợp lệ | 1 |
| Tổng video có lỗi | **34/338** |

304 video còn lại qua kiểm tra cấu trúc này; chưa có nghĩa đã qua visibility/media audit. Không dùng held-out relation distribution để tạo prior hoặc chọn model. Các tên lạ ghi nhận ở lỗi đầu tiên gồm `climbing`, `enclosing`, `getting down on`, `pouring`, `squatting on`, `squeezing`. Có cả trường hợp end bằng `num_frames` và vượt rất xa metadata; chưa đủ bằng chứng áp dụng một phép sửa đồng loạt.

M2 cần đối chiếu preprocessing/upstream artifacts với frame/mask thực, xác minh mapping nhãn và thời gian, rồi tạo **annotation adapter/repair manifest có version và provenance riêng nếu cần**. Giữ nguyên raw annotation, outer split và các lock M0/M1; không âm thầm drop 34 video để train hoặc sửa danh sách 338/62. Khóa fit/internal-val sau khi audit được dữ liệu; chưa tạo feature cache hoặc lệnh train trước đó.

Implementation M1 dùng các tensor CPU dense theo một video: riêng merged logits float64 cần `P × T × 57 × 8` bytes, còn có buffers/probabilities. Đây là reference implementation phục vụ kiểm tra semantics; pair chunking/streaming cần đo trên dữ liệu thật trước khi mở rộng. Graph construction tránh dense `(N×T)²` attention nhưng vẫn có spatial/query pairs theo `T×N²`.

[m1_status.json](provenance/m1_status.json) ghi hashes source/artifacts và các giới hạn; [m1_verification.json](provenance/m1_verification.json) ghi kiểm tra CPU; [m1_demo_result.json](provenance/m1_demo_result.json) lưu các kết quả chính của demo giả. Chưa triển khai loss/trainer, trích features hay GAT layers — các phần đó thuộc M2/M3.
