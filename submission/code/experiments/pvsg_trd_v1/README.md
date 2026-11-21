# PVSG TRD v1 — M0

M0 khóa nguồn dữ liệu, danh sách split và đặc tả thực nghiệm **GT-track Temporal Relation Detection**, không có KG. MLP, Static GAT, ST-GAT và baseline Conv CVPR 2023 chuyển thể sẽ dùng chung protocol `pvsg-gt-trd-tiouv1`. Chưa có parser, evaluator, feature extractor hay trainer; các phần này thuộc milestones sau trong [kế hoạch nghiên cứu](../../docs/research/pvsg_trd_plan.md).

## Split được chọn

Người dùng đã chọn **338 train / 62 test chính thức đã xác minh**. Annotation của tác giả dùng key `val` cho 62 video; thực nghiệm giữ nguyên các ID đó làm **test**, không dùng để chọn checkpoint, threshold hoặc hyperparameter. Internal-validation sẽ tách khoảng 15% từ 338 train, seed 17, có kiểm tra nhóm source/session ở M2; danh sách này hiện chưa khóa.

| Nguồn | Train | Test (`val` upstream) |
|---|---:|---:|
| VidOR | 249 | 40 |
| EPIC-KITCHENS | 44 | 11 |
| Ego4D | 45 | 11 |
| Tổng | **338** | **62** |

[Manifest đã khóa](manifests/pvsg_author_release_338_62.lock.json) chứa đủ 400 video IDs và source map, không chia lại outer split. Annotation có 115 thing + 11 stuff classes, 57 predicates, metadata tổng 149.488 frame ở 5 FPS; số frame này chưa được đối chiếu với media.

Nguồn được pin:

- [Annotation của tác giả trên Hugging Face](https://huggingface.co/datasets/Jingkang/PVSG/resolve/7e5f1ec9fd8f323182e84d990819854bb72da478/pvsg.json), revision `7e5f1ec9fd8f323182e84d990819854bb72da478`, 3.884.637 bytes.
- SHA256 annotation: `2cbc23b060386ccf090475f90cb0282a3e96cefb29c69772f9f8fd916995ba08`.
- SHA256 split IDs: `7437cadd922c9b81236700cad48b46e7bdf3e247d61fd6d0876ae77b04fdf65a`. Cách tính: SHA256 UTF-8 của JSON `{train_ids: sorted(...), heldout_ids: sorted(...)}`, `sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`.

**Sai khác giữa hai bản paper:** [PDF trên CVF, §6](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Panoptic_Video_Scene_Graph_Generation_CVPR_2023_paper.pdf) ghi 360/40; [PDF trên trang tác giả, §6](https://jingkangyang.com/PVSG/static/pdfs/pvsg.pdf) ghi 338/62. Hai revision annotation từ 23/11 và 29/11/2023 có danh sách split giống hệt bản đã chọn. Vì vậy 338/62 có bằng chứng từ chính bản phát hành năm 2023, không chỉ từ các bài về sau. Chưa tìm được manifest ID 360/40 trong các nguồn đã kiểm tra; điều này không chứng minh nó chưa từng tồn tại.

[Template 360/40](manifests/cvf_360_40.unresolved.template.json) chỉ lưu dấu vết mục tiêu cũ, không phải dataset được chọn. Validator từ chối sử dụng template chưa có ID/hash như một manifest đã khóa. Không chuyển 22 video test sang train để tạo lại số lượng trong bản CVF.

## Hợp đồng input/output và evaluation

[protocol.json](protocol.json) là đặc tả máy đọc được; [vocabulary.json](provenance/vocabulary.json) giữ nguyên thứ tự lớp của annotation. Class IDs: `thing + stuff`, 0–125; predicate IDs: 0–56. Track IDs chỉ định danh đối tượng trong video, không làm feature.

| Artifact | Nội dung |
|---|---|
| [video.schema.json](schemas/video.schema.json) | Input GT object classes, track IDs, visibility và tham chiếu masks; không chứa relation targets/captions/QA |
| [targets.schema.json](schemas/targets.schema.json) | Targets riêng: ordered subject/object IDs, predicate ID, danh sách intervals |
| [predictions.schema.json](schemas/predictions.schema.json) | Output cùng định dạng intervals, thêm score `[0,1]` và protocol ID |
| [dataset_manifest.schema.json](schemas/dataset_manifest.schema.json) | Phân biệt template, candidate, locked; nguồn annotation, outer split và internal-validation |
| [examples/](examples/) | Video giả 10 frame, hai tracks, nhiều predicate/đoạn rời; không chứa dữ liệu thật |

Input inference chỉ nhận RGB và GT objects/tracks; không được dùng GT relation pairs, predicates hoặc intervals. Xét tất cả ordered pairs khác nhau có ít nhất một frame đồng hiện. Cùng pair có thể có nhiều predicate; không có graph constraint.

Timeline 0-based, `[start,end]` bao gồm hai đầu. Output là một tuple `(video,s,p,o)` với hợp các intervals; một tuple trùng chỉ tính một lần. Ghép logits của các cửa sổ overlap bằng trung bình trước sigmoid; mặc định threshold 0,5, không nối gap, không bỏ đoạn ngắn. Score là trung bình xác suất trên frame active hợp lệ; rank toàn video sau khi ghép. Khi bằng score, sắp theo subject ID, predicate ID rồi object ID tăng dần.

Metric chính: R/mR@20/50/100 với **temporal set IoU ≥ 0,5**; thêm 0,3 và 0,7. R gộp hits/GT toàn split; mR trung bình recall của các lớp có support. Prediction chỉ bị mask theo visibility, không cắt theo GT relation span. Ví dụ GT active 2/10 frame còn prediction active cả 10 frame cho tIoU = 0,2. GT không còn frame sau visibility filtering phải làm audit thất bại để điều tra, không âm thầm bỏ mẫu.

Đây là **đặc tả riêng cho GT-track TRD**, không tương đương benchmark predicted-track OpenPVSG hoặc thiết lập chỉ thay object labels bằng GT. `calculate_viou` của OpenPVSG đã đọc chỉ xét GT relation frames; không dùng tên metric đó để thay thế tIoU ở đây. Nếu tái lập baseline gốc, dùng namespace riêng `openpvsg-ips-conv-repro-v1`; chuyển thể Conv sang GT tracks dùng `pvsg-conv-gt-adapt-v1`, không gọi là tái lập điểm paper.

Schema M0 kiểm tra cấu trúc; chưa thay thế parser/evaluator. Các bất biến như mọi interval trong `[0,T-1]`, unique track IDs, subject khác object, đúng track/class/visibility phải được kiểm tra đầy đủ ở M1. Validator M0 chỉ kiểm tra các quan hệ này trên fixtures hiện có.

## Kiểm tra CPU

M0 dùng Python 3.12.3, `jsonschema==4.26.0`, `pytest==8.3.5`; không import PyTorch/CUDA. [requirements-cpu-m0.txt](requirements-cpu-m0.txt) ghi dependencies trực tiếp để tái chạy kiểm tra, không phải môi trường train đã khóa.

Chạy từ repository root:

```bash
python experiments/pvsg_trd_v1/validate_m0.py --require-selected
python -m pytest -q experiments/pvsg_trd_v1/tests
```

Nếu annotation local vẫn còn, kiểm tra thêm byte hash, danh sách split/source, vocabulary và FPS:

```bash
python experiments/pvsg_trd_v1/validate_m0.py --require-selected --annotation data/external/pvsg_m0/pvsg.hf_7e5f1ec.json
```

Các lệnh chỉ đọc file local, không tải dữ liệu và không huấn luyện. Annotation/PDF nghiên cứu nằm ngoài Git ở `data/external/pvsg_m0/`; chưa tải archives video/masks hoặc checkpoint. Chạy các kiểm tra artifact không cần annotation local. Chưa lấy phân bố relation của test làm thống kê/prior.

[sources.json](provenance/sources.json) ghi URLs, revisions, hashes và bằng chứng sai khác split. [m0_status.json](provenance/m0_status.json) ghi quyết định người dùng, hash artifacts và các việc còn lại. [verification.json](provenance/verification.json) ghi kết quả kiểm tra CPU. Các hash là dấu vết phiên bản, không phải chữ ký bảo mật; thay đổi đặc tả cần version/provenance mới.

## Bàn giao sang M1/M2

- M1: parser độc lập, decoder và evaluator CPU; fixtures perfect/wrong/overlong/duplicate/empty, multi-label, visibility, ranking và graph batching.
- M2: tải/audit media khi triển khai mốc đó, kiểm tra frame/mask alignment, chốt fit/internal-val chỉ trong 338 train và tạo feature cache. M0 chưa xác nhận media availability hoặc thông lượng.
- Giữ nguyên code/config/manifest đã khóa của pipeline cũ. Phần mới nằm trong thư mục này; nguồn tham khảo repository ở commit `c20812ad52ff3bc6823aae0dd3659eab90caa451`, OpenPVSG ở `b2850b980c272368215a754299986cb904d55820`.
- Chưa có launcher training. Lượt train chính sau này phải được kiểm tra CPU, commit, bàn giao lệnh và theo dõi WandB; người dùng chạy tuần tự một model, không tự chuyển lượt.
