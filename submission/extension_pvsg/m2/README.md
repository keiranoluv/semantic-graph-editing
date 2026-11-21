# M2 — PVSG media, split và feature cache

Phần chuẩn bị dữ liệu/kiểm tra CPU đã hoàn tất; **full visual cache còn chờ người dùng chạy launcher GPU bên dưới**. M1 đã hoàn thành tại commit `f28b9af3`. M2 nằm riêng trong thư mục này, không thay đổi source, protocol hoặc provenance M0/M1 và pipeline thực nghiệm cũ. Không có KG, chưa train relation model.

Đây là dataset variant **`pvsg-gml-valid-support-v1`** cho GT-track TRD, evaluator **`pvsg-gt-trd-tiouv1`**. Dùng bản phát hành PVSG của tác giả năm 2023, outer split **338/62** đã được người dùng chọn. CLIP crops và xử lý support ở đây khác predicted query features/preprocessing của OpenPVSG; chưa phải tái lập điểm benchmark CVPR 2023.

## Dữ liệu và provenance

- Sáu archive chính thức đã tải và kiểm SHA256: **10.792.485.748 bytes** (10,79 GB thập phân), revision `7e5f1ec9fd8f323182e84d990819854bb72da478`. [Media lock](manifests/media.lock.json).
- Có đủ **400 MP4, 149.488 masks**; mọi mask có chỉ số liên tục từ 0, kích thước đúng metadata. Giải mã CPU toàn bộ media, kiểm PTS, ID và visibility. Chỉ giải nén MP4; masks đọc trực tiếp từ ZIP, RGB không ghi thành 150 nghìn ảnh rời.
- [Media audit](provenance/media_audit.json), [thống kê storage/geometry](provenance/data_summary.json), [policy đầu vào](configs/media_policy_v1.json) và [nguồn upstream](provenance/sources.json) ghi kết quả cụ thể. Hash của inputs, targets và từng shard nằm trong [M2 status](provenance/m2_status.json).
- Checkpoint OpenAI CLIP ViT-B/32: **353.976.522 bytes**, SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Visual architecture vendored từ commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, giữ MIT license; chỉ đổi formatting, thêm nguồn và chú thích lint.

Số đo thực: **7.596 tracks**, **1.512.183 visible object observations** (49 observations thiếu RGB), tối đa 38 objects/frame. Geometry nén khoảng **46,27 MB**; MP4 giải nén **8,77 GB**. Visual cache dự kiến khoảng **1,57 GB** gồm fp16 features và row indices, chưa phải dung lượng GPU run đã đo. Audit toàn bộ bằng 4 CPU workers mất **484,46 giây** trên máy hiện tại. Không cache 16.837.384 ordered pair-frame observations thành features N².

Các đường dẫn local được sử dụng:

| Nội dung | Đường dẫn từ repo root |
|---|---|
| Raw annotation bất biến | `data/external/pvsg_m0/pvsg.hf_7e5f1ec.json` |
| Archive / MP4 / index | `data/external/pvsg_media_v1/` |
| Inputs / geometry / targets / ledger chính thức M2 | `data/external/pvsg_m2_cache_v4/` |
| Checkpoint | `data/external/pvsg_clip_v1/ViT-B-32.pt` |
| Visual cache sau khi chạy GPU | `data/external/pvsg_clip_cache_v1/` |

Các cache audit thử nghiệm `pvsg_m2_cache_v1`, `v2`, `v3` là bản debug, **không dùng để train**. Bản v4 giải quyết việc đọc ZIP đồng thời, mask ID thiếu nhãn và gap PTS. Raw archive/annotation không bị sửa.

## Các sai khác media cần giữ trong báo cáo

Có 7 video mà header MP4 và mask count khác nhau. Sau giải mã, 5 video vẫn có đủ RGB tương ứng mask; header không thể dùng thay số frame thực. Hai video còn lại thiếu tổng 4 RGB frames: `1019_3004044251` thiếu frame 111; `ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6` thiếu frames 411–413, frame cuối có PTS 82,8 giây ứng với mask 414. Ghép bằng relative PTS trên lưới 5 FPS, không dồn frame 414 vào vị trí 411. Kiểm tra riêng overlay phần cuối được lưu trong provenance. Không suy đoán một offset mới từ nhãn relation.

Giữ nguyên timeline masks/annotation; tại RGB thiếu, giữ geometry/visibility GT, visual vector bằng 0 và `visual_available=false`. M3 phải dùng cờ này như một input, không nhầm vector thiếu với crop đen hợp lệ. Code cũng hỗ trợ ghi nhận và bỏ RGB tail không có mask nếu gặp; số tail thực tế của bản phát hành này được ghi trong media audit.

4 video có vùng nonzero mask ID không nằm trong object table. Giữ các object có nhãn, bỏ riêng vùng không có nhãn khỏi node inputs và ghi ID/số frame/pixel; không tự gán class hoặc bỏ cả video. Cách đọc tube này phù hợp `get_gt_mask_tubes_one_video` của OpenPVSG, **không** phải khẳng định segmentation loader upstream có thể xử lý các ID đó.

Overlay kiểm tra đầu/giữa/cuối trên 12 video train lấy cố định (4 mỗi nguồn), cộng 7 ca header khác biệt: **19 video / 57 frames**. [Biên bản kiểm tra trực quan](provenance/visual_review.json) và contact sheets đi kèm. Đây là kiểm tra mẫu về alignment, không chứng nhận chất lượng toàn bộ mask hoặc semantic labels.

## Split và targets có version riêng

Internal split đã khóa trong [dataset.lock.json](manifests/dataset.lock.json):

| Nguồn | Fit | Internal-val | Held-out test |
|---|---:|---:|---:|
| VidOR | 212 | 37 | 40 |
| EPIC-KITCHENS | 37 | 7 | 11 |
| Ego4D | 38 | 7 | 11 |
| Tổng | **287** | **51** | **62** |

Chỉ dùng official train IDs để chọn internal-val; phân tầng nguồn, group theo source ID suy ra từ tên clip, seed 17, chọn gần 15% bằng thứ tự hash/subset sum. Fit và internal-val không chung nhóm. Grouping này không thay thế metadata session đầy đủ. Official outer split có **4 nhóm source Ego4D chung train/test**; giữ nguyên split người dùng đã chọn và ghi giới hạn này trong [group audit](provenance/groups.json). Chung source UUID chưa chứng minh cùng đoạn frame.

`preparation.py` xử lý nhất quán cả 400 video, không dùng test accuracy để chọn policy:

1. Giữ vocabulary 57 predicates; quarantine row ngoài ontology, giống bộ lọc `PVSGRelationAnnotation.__getitem__` upstream. Không đổi từ đồng nghĩa thành nhãn mới.
2. Quarantine self-pair hoặc track reference không tồn tại.
3. Lấy giao integer intervals **inclusive** với timeline mask và visibility của cả hai tracks, hợp nhất các đoạn trùng. Row không còn support vào ledger, không sửa FPS hoặc nhân/chia timestamps để đoán nhãn.
4. Với `(subject, predicate, object)` có interval lỗi/empty support, bỏ **negative supervision** của key đó; các positive hợp lệ vẫn được học. M3 dùng `rasterize_adapted_targets(...)["supervision_mask"]` cho mọi baseline. Không gọi loader strict M1 trực tiếp trên raw annotation hoặc dùng toàn bộ zero labels làm negative.

Từ 4.587 raw relation rows, còn **4.384 unique GT tuples** sau adaptation: quarantine 39 rows ngoài ontology, 2 self-pair/track lỗi và 162 rows không còn covisible support; 52 rows hợp lệ được lấy giao time bounds, 2.493 rows thay đổi do visibility/union. Có 14.242 interval segments sau khi tách theo visibility; không so số này trực tiếp với 6.035 raw interval entries.

[Annotation summary](provenance/annotation_summary.json) ghi số row mỗi loại và ví dụ; ledger đầy đủ là `annotation_adaptation.json` trong cache v4. Thống kê predicate support chỉ dùng **fit**. Targets/evaluation support đã thay đổi so với raw annotation: phải báo policy ID và denominator cùng metrics; không đặt điểm này ngang hàng end-to-end PVSG. Quarantine không đồng nghĩa đã phục hồi được nhãn gốc.

## Input cache

Geometry shard theo video (`allow_pickle=False`):

- `track_ids[N]`, `class_ids[N]`; giữ thứ tự sorted track ID.
- `boxes[N,T,4]`: pixel bbox với right/bottom exclusive.
- `geometry[N,T,6]`: normalized cx, cy, width, height, box area, mask area.
- `visible[N,T]`, `rgb_available[T]`; dữ liệu thiếu để 0 và có cờ riêng.
- `motion[N,T,4]`, `delta_frames[N,T]`: sai khác 4 kênh geometry đầu / số frame giữa các quan sát hợp lệ. Không tính sai khác qua padding như thể object đứng yên.

Visual shard: `features[M,512]` fp16 L2-normalized, `frame_ids[M]`, `track_ids[M]`, `visual_available[M]`. M là số visible object observations, thứ tự frame trước rồi track ID; không cache N² pair features. Crop từ bbox mask, vùng ngoài instance tô RGB `[123,117,104]`, resize bicubic cạnh ngắn 224, center crop, CLIP normalization. [Config đầy đủ](configs/clip_masked_v1.json).

Không có predicate/pair/span GT trong extractor. Trích cả 400 video bằng encoder đóng băng không học statistics từ test. Features/geometry dùng chung cho MLP, Static GAT, ST-GAT và Conv baseline về sau.

Mỗi video ghi NPZ và metadata atomically; fingerprint chứa code/config/runtime/precision/batch/input hashes. Resume chỉ bỏ qua shard khớp hash, không ghi đè cache khác version. Khóa file ngăn hai extractor cùng ghi vào output. `summary.json` chỉ đánh dấu `complete_cohort=true` khi đủ 400 ID.

## Môi trường và kiểm tra CPU

Interpreter local: `data/external/pvsg_m2_cpu_env/bin/python` (Python 3.12.3). Đây là overlay venv: đọc PyTorch/Pillow/pytest từ `myenv` qua `.pth`, cài riêng PyAV 16.0.1, SciPy 1.15.3, NumPy 2.1.0. Không sửa environment cũ. PyAV có FFmpeg libraries đi kèm; không cần system `ffmpeg`, OpenCV, MMDetection hoặc PyG trong M2.

Torch đang dùng `2.9.1+cu128`. torchvision hiện có không tương thích với bản torch này, nên extractor dùng visual module đã pin và Pillow; không import torchvision. Clean environment trên máy khác có thể cài [requirements-feature-m2.txt](requirements-feature-m2.txt) bằng pip trong venv mới; runtime versions được preflight kiểm lại. Overlay hiện tại chưa phải container portable độc lập.

Kiểm tra có ý nghĩa: download resume/hash failure; tách video ID trong ZIP; mask geometry/visibility; gap PTS và tail RGB; group split; ledger và ignore-negative mask; crop preprocessing; feature row alignment/missing RGB; cache resume và từ chối provenance khác. M0/M1 tests chạy lại cùng M2: **110 tests đạt** (97 M0/M1 + 13 M2).

[CPU probe thực](provenance/clip_cpu_probe.json) dùng 3 crop, mỗi nguồn một crop: output 3×512, norm sai số dưới 1e-6, hai lần forward khớp; model frozen/eval; CUDA chưa khởi tạo. Đây là smoke check CPU, không suy ra throughput hay VRAM GPU.

Chạy kiểm tra lại, từ repo root:

```bash
data/external/pvsg_m2_cpu_env/bin/python -m pytest -q extension_pvsg/tests extension_pvsg/m2
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.preflight
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.probe_clip_cpu --output data/external/pvsg_clip_v1/cpu_probe_repeat.json
```

Các entrypoints tái tạo dữ liệu (không cần chạy lại trên workspace hiện tại):

```bash
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.download
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.download --manifest extension_pvsg/m2/manifests/encoder.lock.json --directory data/external/pvsg_clip_v1
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.media
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.audit_media
data/external/pvsg_m2_cpu_env/bin/python -m extension_pvsg.m2.preparation
```

Thời gian chạy nằm trong provenance nên tái tạo audit sẽ có hash report khác; cần snapshot provenance mới, không sửa status đã commit để trộn kết quả. Raw asset/split và numeric preprocessing vẫn được pin.

## Lệnh GPU người dùng chạy để hoàn tất M2

Code phải đã commit trước khi chạy; launcher tự kiểm điều này và toàn bộ lock trên CPU:

```bash
bash extension_pvsg/m2/launch_extract.sh
```

- **Model:** OpenAI CLIP ViT-B/32 visual encoder, frozen/eval, CUDA fp16 autocast, output fp16 512 chiều.
- **Batch:** 64 object crops; **seed:** 17; GPU mặc định 0, đổi bằng `PVSG_GPU=1 bash ...` khi cần. Batch 64 là cấu hình xuất phát, chưa benchmark VRAM/thời gian trên GPU này.
- **Phạm vi:** một encoder xử lý tuần tự 400 video, dùng cùng feature config; output `data/external/pvsg_clip_cache_v1/`. Không có optimizer, epoch hoặc training.
- **Resume:** chạy lại cùng lệnh, xác minh/bỏ qua video đã hoàn tất; video dở được làm lại. Nếu OOM cần batch khác, dùng output version mới, không sửa/trộn cache đang có.
- **Queue:** chạy tiếp các video còn lại trong cohort rồi dừng; **không tự train MLP/GAT, không chạy model/seed khác**. Full training M3 về sau phải có WandB theo quy ước dự án; lượt extraction này lưu provenance local.

Để nghiệm thu nốt M2: `summary.json` có `complete_cohort=true`, 400 video, rows bằng audit, không có shard lỗi; ghi storage/throughput GPU thực tế và kiểm một mẫu feature→(video,frame,track). Hiện chưa có kết quả GPU hoặc benchmark TRD.
