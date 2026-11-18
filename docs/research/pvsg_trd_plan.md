# Kế hoạch mở rộng Semantic Graph Editing sang PVSG Temporal Relation Detection

Ngày khảo sát và cập nhật M0/M1/M2/M3: **18/11/2026 (Asia/Bangkok)**. Repository ban đầu được kiểm tra tại commit `c20812ad52ff3bc6823aae0dd3659eab90caa451`, M0 khóa ở `c576cfdcd4a51a7f3c76d6260d89c4695b0c0118`, M1 hoàn thành tại `f28b9af3`. M2 đã tải media/checkpoint, audit CPU và chuẩn bị feature extractor; **full visual cache còn chờ người dùng chạy GPU**. Chưa training hoặc sửa pipeline cũ. Các con số benchmark dưới đây là kết quả công bố, không phải kết quả tái lập của đồ án. Xem [snapshot M0](../../experiments/pvsg_trd_v1/README.md), [bàn giao M1](../../experiments/pvsg_trd_v1/README_M1.md), [bàn giao M2](../../experiments/pvsg_trd_v1/m2/README.md) và [code/CPU checks M3](../../experiments/pvsg_trd_v1/m3/README.md).

## 1. Kết luận và phạm vi

**Nên triển khai một thực nghiệm GT-track TRD độc lập:** cho sẵn masks, nhãn đối tượng và track IDs; dùng backbone thị giác đóng băng; huấn luyện MLP, Static GAT và Spatio-Temporal GAT để dự đoán predicate cùng thời gian tồn tại. **Không triển khai KG cho nhánh PVSG**, theo phạm vi đã thống nhất với người dùng. Phần bổ sung ưu tiên ablation và tái lập một baseline CVPR 2023 nếu nguồn dữ liệu, checkpoint và protocol cho phép. Không đặt mục tiêu vượt SOTA hoặc tái huấn luyện segmentation/tracking.

**Lựa chọn dataset/split đã thống nhất sau M0:** dùng bản PVSG do tác giả phát hành từ tháng 11/2023, **338 video train / 62 video test**. Đã xác minh đầy đủ ID từ annotation, đối chiếu hai revision lịch sử và PDF trên trang tác giả. Người dùng đã chọn split này thay mục tiêu 360/40 ghi trong bản CVF nhưng chưa tìm được manifest ID tương ứng. Không chia lại outer split; `val` upstream giữ vai trò test. [Annotation đã pin][hf-annotation], [PDF tác giả][pvsg-author-paper].

Theo [đề bài](../../Project01_Graph_Editing.pdf), trang 1–4, phần được huấn luyện là mô hình đồ thị; baseline phi đồ thị và ablation cạnh đều phù hợp mục tiêu môn học. Quyết định bỏ KG chỉ áp dụng cho thực nghiệm PVSG mở rộng; yêu cầu tri thức của phần ảnh tĩnh vẫn được ghi nhận ở phạm vi đồ án gốc. PVSG cung cấp temporal scene graph supervision trực tiếp, thuận tiện hơn việc tự tạo nhãn quan hệ từ DAVIS/YouTube-VOS. Tuy nhiên:

- Đây là mở rộng **nhiệm vụ 1 sang video**. TRD chưa thực hiện chỉnh sửa video hoặc chấm tính nhất quán trước/sau chỉnh sửa như hướng video trong đề bài.
- Không thay thế các yêu cầu nhiệm vụ 2/3, cũng không thay thế phần ghi nhận khó khăn và thay đổi dataset của nhiệm vụ 2 task B.
- Việc khảo sát SegPVSG 2026 không đồng nghĩa đã tái lập một phương pháp 2026. Báo cáo cần phân biệt nghiên cứu tài liệu, mô hình tự triển khai đơn giản và phương pháp đã tái lập.

### Input và output

Với video có `T` frame trên lưới annotation 5 FPS, input gồm RGB tùy cấu hình, tập track `i`, class `c_i`, mask `M_i(t)`, cờ hiện diện `a_i(t)` và đặc trưng đóng băng `v_i(t)`. Track ID chỉ dùng để nối cùng đối tượng qua thời gian; **không đưa giá trị ID vào embedding học được**.

Output cho mỗi video:

```text
(subject_track_id, predicate, object_track_id,
 intervals=[(start_frame, end_frame), ...], score)
```

Một bộ `(subject, predicate, object)` có thể xuất hiện nhiều đoạn rời nhau, và một cặp có thể có nhiều predicate đồng thời. Vì vậy, output trung gian là tensor **multi-label** `logits[P,T,57]`, không phải một nhãn softmax cho toàn video. `P` là số cặp có hướng được xét; các chiều thời gian/track thiếu được mask.

Protocol mặc định là **offline**: được nhìn cả cửa sổ trước và sau frame cần dự đoán. Không gọi đây là dự đoán tương lai hoặc xử lý causal thời gian thực.

## 2. Kiểm tra repository: tái sử dụng ở mức nào?

| Thành phần đã kiểm tra | Có thể dùng lại | Phần cần viết riêng cho PVSG |
|---|---|---|
| [`src/models/mlp.py`](../../src/models/mlp.py), [`mlp_semantic.py`](../../src/models/mlp_semantic.py) | MLP cho cặp có thứ tự; embedding class; kích thước đầu ra cấu hình được | Tensor theo thời gian, pair geometry, 57 sigmoid logits, mask loss và decoder intervals; không dùng checkpoint VG150 |
| [`src/models/gat.py`](../../src/models/gat.py) | `GATPredicateClassifier`, `encode_nodes`, residual + LayerNorm, `score_pairs`; topology độc lập nhãn quan hệ | Static GAT áp dụng từng frame; ST-GAT cần loại cạnh spatial/temporal và positional information riêng |
| [`src/graph/builder.py`](../../src/graph/builder.py) | All ordered pairs, ánh xạ instance ID → node index, tách cạnh message passing và GT | Builder hiện dành cho một ảnh; `edge_label` chiếu về single-label và đặt `-1` cho nhiều predicate. Không dùng phép chiếu này cho PVSG |
| [`src/graph/gnn_batching.py`](../../src/graph/gnn_batching.py), [`src/datasets/batching.py`](../../src/datasets/batching.py) | Tham khảo batching và quản lý RNG | Collator cho video/cửa sổ, visibility, track offsets; ngăn nối cạnh giữa hai video |
| [`src/features/geometry.py`](../../src/features/geometry.py) | Công thức geometry chuẩn hóa | Tạo box từ mask; thêm mask area/centroid và motion; khóa quy ước pixel |
| [`src/features/visual.py`](../../src/features/visual.py), [`scripts/cache_vg150_regions.py`](../../scripts/cache_vg150_regions.py) | Cách đóng băng encoder, cache và kiểm tra provenance | Cache hiện gắn HDF5 box rows, image IDs và VG150; cần khóa mới `(video_id, frame_id, track_id)` |
| [`src/knowledge/`](../../src/knowledge/), [`gat_kg.py`](../../src/models/gat_kg.py), [`gat_pair_kg.py`](../../src/models/gat_pair_kg.py), [`gat_kg_fusion.py`](../../src/models/gat_kg_fusion.py) | Đã khảo sát: linking WordNet/ConceptNet, TransE/ComplEx/RotatE, export và fusion | **Ngoài phạm vi PVSG đã chốt**. Giữ ghi nhận khảo sát; không tạo mapping, embedding, exporter hoặc model KG cho PVSG. Các artifact hiện có gắn vocabulary VG150, không chuyển trực tiếp |
| [`src/evaluation/scene_graph.py`](../../src/evaluation/scene_graph.py), [`reference_predcls.py`](../../src/evaluation/reference_predcls.py), [`protocols.py`](../../src/evaluation/protocols.py) | Cách version protocol, xuất support/hits theo lớp, ranking và kiểm tra input | Evaluator ảnh dùng softmax/background, matching box hoặc pair và image-macro recall; không thay được evaluator video/tIoU bằng đổi `K` |
| [`src/training/trainer.py`](../../src/training/trainer.py) và các runner versioned | Tham khảo seed, resume, artifact logging | Training hiện dùng cross-entropy; cần trainer multi-label, window sampling, validation theo video và WandB riêng |

**Quyết định:** tái sử dụng các khối toán học và kinh nghiệm kiểm soát thực nghiệm, không cố đưa PVSG qua schema VG150. Ưu tiên adapter/import chỉ đọc cho module thuần có interface phù hợp; nếu cần đổi semantics thì tạo implementation versioned riêng, ghi đường dẫn/commit nguồn tham khảo. Không kế thừa checkpoint classifier 50 lớp hoặc cache cũ chỉ vì kích thước tensor giống nhau.

[`tools/level1_test_tail_zero_v1.py`](../../tools/level1_test_tail_zero_v1.py) thực sự hash toàn bộ `src/**/*.py` và `scripts/*.py`. Cùng với [AGENTS.md](../../AGENTS.md), đây là lý do phần mở rộng nên ở `experiments/pvsg_trd_v1/`; không thêm/sửa `src/`, manifest/config đã khóa hoặc notebook cá nhân.

## 3. Dataset OpenPVSG

### Quy mô và cấu trúc

PVSG gồm **400 video**, khoảng **150 nghìn frame**, **126 object classes**, **57 predicates**; 289 video VidOR, 55 EPIC-KITCHENS và 56 Ego4D. Video dài trung bình 76,5 giây, annotation ở 5 FPS. Có góc nhìn thứ nhất/thứ ba, thing/stuff và quan hệ nền–đối tượng. [Trang dự án][pvsg-page], [OpenPVSG README][openpvsg-readme], [bài CVPR][pvsg-paper].

```text
data/pvsg/                       # vị trí dự kiến, chưa tạo/tải
├── pvsg.json
├── ego4d/
│   ├── videos/
│   ├── frames/<video_id>/...
│   └── masks/<video_id>/0000.png, ...
├── epic_kitchen/...
└── vidor/...
```

Code loader xác định vocabulary bằng `objects.thing + objects.stuff`; `relations` là vocabulary predicate; `data` chứa annotation theo `video_id`; `split[source][split_name]` chứa danh sách video. Bản minh họa schema sau **không phải mẫu tải từ dataset**:

```json
{
  "objects": {"thing": ["..."], "stuff": ["..."]},
  "relations": ["..."],
  "split": {"vidor": {"train": ["..."], "val": ["..."]}},
  "data": [{
    "video_id": "example",
    "objects": [{"object_id": 7, "category": "adult"}],
    "relations": [[7, 12, "holding", [[10, 25], [40, 51]]]]
  }]
}
```

`object_id` là định danh instance trong video, không phải class ID. Loader GT đọc `panoptic_mask == object_id` để lấy mask của track; category trong JSON được ánh xạ qua vocabulary. Có sẵn `get_gt_mask_tubes_one_video`. Khoảng `[start,end]` trong evaluator được duyệt **bao gồm cả hai đầu**. Predicted tracking file có bước đổi frame index `fid - 1`; đây là điểm dễ sai khi tích hợp. [Annotation/tube utilities][openpvsg-matching], [relation dataset loader][openpvsg-dataset].

### Split đã khóa: bản phát hành của tác giả, 338 train / 62 test

| Nguồn | Split được mô tả | Cách xử lý trong đồ án |
|---|---|---|
| PDF CVF CVPR 2023, §6 | 360 train / 40 test | Ghi nhận sai khác lịch sử; chưa xác minh được danh sách ID tương ứng |
| PDF trên trang tác giả, §6, và annotation tháng 11/2023 | 338 train / 62 test | **Đã chọn và khóa ID/hash theo xác nhận của người dùng** |
| SegPVSG, §4.1 | 338 train / 62 test | Cùng số lượng; không tự chứng minh mọi evaluator/artifact đều giống nhau |
| OpenPVSG loader và test script | Đọc `train` và `val` theo từng nguồn | `val` được test script sử dụng để đánh giá; không coi đó đương nhiên là tập tuning |

Nguồn: [CVF PDF][pvsg-paper], [PDF tác giả][pvsg-author-paper], [annotation đã pin][hf-annotation], [SegPVSG PDF][seg-paper], [loader][openpvsg-dataset], [full evaluation][openpvsg-full-test]. M0 đã đọc `pvsg.json` trực tiếp, kiểm tra 400 ID duy nhất và train/test không giao nhau. Không dùng preview tự động của Hugging Face để suy ra split.

Artifact được chọn: revision `7e5f1ec9fd8f323182e84d990819854bb72da478`, 3.884.637 bytes, annotation SHA256 `2cbc23b060386ccf090475f90cb0282a3e96cefb29c69772f9f8fd916995ba08`. Revision 23/11/2023 và 29/11/2023 có danh sách split và vocabulary giống hệt bản này. Như vậy 338/62 đã có trong bản phát hành năm 2023, không phải chỉ xuất hiện ở công trình sau. [Provenance M0](../../experiments/pvsg_trd_v1/provenance/sources.json).

| Nguồn video | Train | Test (`val`) |
|---|---:|---:|
| VidOR | 249 | 40 |
| EPIC-KITCHENS | 44 | 11 |
| Ego4D | 45 | 11 |
| Tổng | **338** | **62** |

[Manifest M0](../../experiments/pvsg_trd_v1/manifests/pvsg_author_release_338_62.lock.json) khóa đủ ID/source và SHA256 danh sách split `7437cadd922c9b81236700cad48b46e7bdf3e247d61fd6d0876ae77b04fdf65a`. Template 360/40 chỉ lưu dấu vết mục tiêu chưa xác minh, không được chọn để train. Không dựng lại 360/40 hoặc chuyển 22 video test sang train. Chưa tìm được manifest 360/40 trong nguồn đã kiểm tra không có nghĩa nó chưa từng tồn tại.

M0 xác minh metadata tổng **149.488 frame**, 115 thing + 11 stuff classes và 57 predicates, tất cả video ở 5 FPS. M2 đã đối chiếu media thực; số visible tracks/triplets/intervals nằm trong [data summary](../../experiments/pvsg_trd_v1/m2/provenance/data_summary.json), phân bố predicate chỉ thống kê trên fit. Các số “relation” trong paper có thể đếm bộ ba hợp nhất hoặc các đoạn; báo cáo cả hai, không dùng thay nhau. M0 chưa dùng phân bố relation của test để tạo prior hoặc quyết định model.

**Phát hiện từ parser M1:** kiểm tra cấu trúc raw annotation trên 338 train video phát hiện 34 video có lỗi: 17 lỗi predicate ngoài vocabulary, 16 lỗi interval vượt `num_frames`, 1 lỗi self-pair/track reference; mỗi video chỉ báo lỗi đầu tiên. [Report audit train](../../experiments/pvsg_trd_v1/provenance/m1_train_annotation_audit.json). 304 video qua kiểm tra cấu trúc chưa đồng nghĩa media/visibility hợp lệ. M2 đã đối chiếu upstream và media, thêm policy riêng `pvsg-gml-valid-support-v1`: quarantine predicate ngoài ontology/self-pair/track reference lỗi; lấy giao intervals với mask timeline và covisibility; quarantine empty support; bỏ negative supervision của key có thời gian lỗi nhưng giữ positives hợp lệ. Có ledger đầy đủ, raw annotation/M1 strict parser/400 IDs/outer split giữ nguyên. Đây là dataset variant GML, không phải khôi phục nhãn gốc hoặc tái lập benchmark gốc. Feature cache GPU chưa hoàn tất nên chưa train. [Policy và ledger summary](../../experiments/pvsg_trd_v1/m2/provenance/annotation_summary.json).

### Dùng GT để chỉ học relation prediction có khả thi không?

**Có.** Mask và track ID cung cấp vùng và correspondence; class GT cung cấp semantic input. Có thể bỏ hoàn toàn bước train segmentation và tracker:

1. Đọc masks để tạo geometry và visibility, nối instance theo track ID.
2. Trích đặc trưng từ backbone pretrained đóng băng, dùng crop theo GT box hoặc masked pooling. Cache dùng chung cho tất cả mô hình.
3. Chỉ học semantic projection, GNN và relation heads bằng relation annotations của train.
4. Test chỉ nhận RGB/GT objects/tracks, không nhận GT predicates, GT relation pairs hay GT intervals.

Đây là **oracle objects/tracks**, tương tự PredCls mở rộng sang video; không phải PVSG end-to-end. `PVSGRelationDataset` mặc định đọc `relations.pickle` và feature tubes của pipeline dự đoán: không phải loader GT-track dùng ngay. Cần adapter riêng từ raw annotations/masks. [Code loader][openpvsg-dataset], [GT tube reader][openpvsg-matching].

## 4. Benchmark và các công trình liên quan

### Phương pháp, protocol và code

| Công trình | Kiến trúc chính | Đầu vào object ở benchmark chính | Code và mức sẵn sàng quan sát được |
|---|---|---|---|
| **PVSG, CVPR 2023** | Mask2Former + UniTrack (IPS+T) hoặc VPS; relation head Vanilla, handcrafted temporal filter, 1D Conv, Transformer | Mask tubes/nhãn dự đoán; hai giai đoạn | [OpenPVSG][openpvsg-readme]: loader, train/test, model zoo; dependencies cũ |
| **MCL, AAAI 2025** | Motion-aware contrastive learning trên tube features; shuffle negatives và triplet negatives; optimal transport để so tube; ghép Conv/Transformer | Predicted IPS+T hoặc VPS; không phải GT-track | [Repo do tác giả liên kết][mcl-code]. README công khai các wrapper/placeholder cho segmentation, optical flow và giả định schema; chưa phải pipeline tái lập đã được kiểm chứng ở máy này |
| **UNO, WACV 2026** | Object/relation slots, temporal consistency và dynamic triplet association; hỗ trợ box/mask trong một kiến trúc | One-stage end-to-end, không cho sẵn GT tracks | [Fsoft-AIC/UNO][uno-code]. README còn ghi đang phát triển, hướng dẫn tập trung DSGG và checkpoint link để trống; chưa xác nhận recipe PVSG đầy đủ |
| **SegPVSG, ICML 2026** | TempFocusNet định vị đoạn rồi nhận dạng; thêm RGVA tạo dữ liệu cho relation hiếm | Benchmark chính dùng predicted IPS+T/VPS; không phải GT-track | [ticatt/SegPVSG][seg-code]. Đã phát hành TFN, eval và demo checkpoints; README cảnh báo sai khác metric theo code/environment; chưa xác nhận toàn bộ quy trình RGVA tái lập được từ release |

Nguồn kiến trúc: [PVSG][pvsg-paper], [MCL][mcl-paper], [UNO][uno-paper], [SegPVSG][seg-venue]. Với đồ án, học thiết kế temporal modeling và localization; không đưa optimal transport, slot learning hoặc generative augmentation vào scope bắt buộc.

### Kết quả trên predicted tracks — đơn vị %

Mỗi ô là **R@K / mR@K**. Đây là các hàng trong bảng công bố MCL/UNO/SegPVSG, ở ngưỡng được tác giả gọi là **vIoU = 0,5**; chưa chứng nhận evaluator, checkpoint và split hoàn toàn giống nhau qua mọi release.

| Phương pháp | Nhánh/backbone | @20 | @50 | @100 | Nguồn bảng |
|---|---|---:|---:|---:|---|
| PVSG Conv | IPS+T, R50 | 3.88 / 2.55 | 5.24 / 3.29 | 6.71 / 5.36 | MCL, Bảng 1 |
| PVSG Transformer | IPS+T, R50 | 3.88 / 2.81 | 5.66 / 4.12 | 6.18 / 4.44 | MCL, Bảng 1 |
| MCL Conv | IPS+T, R50 | 4.51 / 3.56 | 6.08 / 4.38 | 7.76 / 5.86 | MCL, Bảng 1 |
| MCL Transformer | IPS+T, R50 | 3.98 / 2.98 | 5.97 / 4.20 | 7.44 / 5.15 | MCL, Bảng 1 |
| UNO | One-stage, R50 | 6.23 / 5.60 | 7.37 / 6.84 | 8.65 / 8.21 | UNO, Bảng 3 |
| UNO | One-stage, ViT-L/14 | 9.44 / 8.25 | 10.83 / 9.72 | 11.59 / 10.86 | UNO, Bảng 3 |
| TFN, không RGVA | IPS+T, R50 | 5.14 / 6.67 | 9.12 / 10.15 | 12.58 / 12.87 | SegPVSG, Bảng 1 |
| SegPVSG | IPS+T, R50, có dữ liệu sinh | 5.45 / 9.13 | 9.54 / 12.74 | 13.63 / 15.91 | SegPVSG, Bảng 1 |
| MCL Conv | VPS, R50 | 0.84 / 0.98 | 1.26 / 1.22 | 1.26 / 1.22 | MCL, Bảng 1 |
| SegPVSG | VPS, R50, có dữ liệu sinh | 1.78 / 2.26 | 2.52 / 2.67 | 3.04 / 4.82 | SegPVSG, Bảng 1 |

Nguồn số liệu: [MCL PDF][mcl-paper], [UNO PDF][uno-paper], [SegPVSG PDF][seg-paper]. Không dùng các số PredCls rất cao trong Bảng 2 UNO: bảng đó thuộc **Action Genome**, không phải PVSG.

Một số mốc tại **vIoU = 0,1**, cũng dùng predicted objects/tracks:

| Phương pháp | @20 | @50 | @100 |
|---|---:|---:|---:|
| MCL Conv, IPS+T R50 | 11.43 / 9.57 | 17.30 / 13.13 | 22.85 / 17.48 |
| UNO ViT-L/14 | 17.54 / 14.81 | 23.82 / 18.31 | 27.32 / 22.14 |
| SegPVSG IPS+T R50 | 11.95 / 16.54 | 20.13 / 22.98 | 28.72 / 28.92 |

Nguồn tương ứng: [MCL, Bảng 1][mcl-paper]; [UNO, Bảng 3][uno-paper]; [SegPVSG, Bảng 1][seg-paper].

**Mốc công bố ở bản CVF, khác mô tả split đã chọn:** CVPR 2023 Bảng 2 ghi IPS+T Transformer `4.02/1.75`, `4.41/1.86`, `4.88/2.03` tại K=20/50/100. Bản PDF này dùng mô tả metric có soft temporal recall và split 360/40. Không thay thế các số này bằng hàng “PVSG Transformer” trong bảng benchmark về sau, hoặc tính mức tăng giữa hai bảng như cùng protocol. Split 338/62 đã khóa cũng không làm GT-track TRD so sánh trực tiếp được với predicted-track PVSG; evaluator và object inputs vẫn khác. [CVF PDF][pvsg-paper].

**Cách diễn giải SOTA:** trong các hàng được khảo sát, UNO ViT-L có R@20/R@50 cao nhất, SegPVSG IPS+T có R@100 và mR tốt hơn. Không có một mô hình thắng mọi metric; khác backbone và dữ liệu augmentation cũng phải được ghi rõ. Đây là mốc tham khảo trong bốn công trình yêu cầu, **không khẳng định một leaderboard đầy đủ cho toàn bộ công trình năm 2026**. Các baseline GML GT-track cần so với nhau trên cùng dữ liệu/features/evaluator, không tính “vượt SOTA” từ bảng trên.

### Đọc đúng chữ “GT” trong SegPVSG

| Thiết lập, Bảng 13 phụ lục | R/mR@20 | R/mR@50 | R/mR@100 |
|---|---:|---:|---:|
| TFN + GT | 13.10 / 15.10 | 23.38 / 24.32 | 32.39 / 31.75 |
| SegPVSG + GT | 13.63 / 15.62 | 26.31 / 27.80 | 41.09 / 39.36 |

Phụ lục §B.8 mô tả **thay predicted entity labels bằng GT labels khi evaluation**, trong thiết lập IPS+T. Không có căn cứ gọi bảng này là cung cấp trọn bộ GT masks + GT track IDs. Bảng 12 “Ground-Truth” lại nói về **interval supervision lúc train**, không phải GT interval ở test. Vì vậy, cả hai đều không là benchmark GT-track TRD trực tiếp cho đề xuất này. [SegPVSG, Bảng 12–13 và §B.8][seg-paper].

## 5. Evaluation: phần cần khóa trước khi train

### Ba khái niệm không được nhập làm một

1. **Temporal IoU:** độ giao/hợp của các frame quan hệ đang hoạt động. Với tập frame dự đoán `P` và GT `G`, `tIoU = |P ∩ G| / |P ∪ G|`. Tập có thể là hợp nhiều interval rời nhau.
2. **Volumetric mask-tube IoU theo định nghĩa thông thường:** `Σ_t |M_pred(t) ∩ M_gt(t)| / Σ_t |M_pred(t) ∪ M_gt(t)|`, có xét support thời gian. Phải xét riêng subject và object; không bằng tIoU nói chung, kể cả khi diện tích object biến thiên theo thời gian.
3. **Metric thực thi của OpenPVSG:** phải đọc đúng script và version, không suy ra từ tên `calculate_viou`.

### Phát hiện từ việc đọc code upstream

Tại OpenPVSG commit `b2850b980c272368215a754299986cb904d55820`:

- [`rel_test.py`][openpvsg-rel-test] match tuple pair/predicate rồi dùng `calculate_iou` trên binary temporal spans, ngưỡng 0,5. Nhưng GT ở đây đến từ relation set đã ghép với predicted tubes; không mặc nhiên đại diện toàn bộ GT gốc.
- [`rel_test_full.py`][openpvsg-full-test] match class subject/object/predicate rồi gọi [`calculate_viou`][openpvsg-metrics]. Hàm chỉ duyệt GT relation frames, kiểm tra mask IoU của **cả hai object ≥ 0,5** mỗi frame, rồi giao với predicted span. Các predicted-positive frame ngoài GT interval không vào union này. Với masks hoàn hảo, kết quả trở thành tỷ lệ phủ GT, không phải tIoU đầy đủ.
- Ví dụ suy ra trực tiếp từ code: GT hoạt động 2 frame trong video 10 frame; dự đoán hoạt động cả 10 frame, masks/nhãn đúng. Hàm full cho `1.0`, còn tIoU thật là `0.2`. Đây là lý do không dùng riêng full evaluator để kết luận mô hình định vị thời gian tốt.
- Full script có thể cộng `weak_hit` nhiều lần cho cùng GT ở ngưỡng 0,1 trước khi gặp match ≥0,5 và `break`; cần fixture kiểm tra. Tên `weak` cũng khác ý nghĩa giữa hai script.
- [`calculate_final_metrics`][openpvsg-metrics] dùng recall gộp tổng hits/tổng GT, mR trên những predicate có GT trong split. Khác image-macro và mean trên cố định 50 predicates của evaluator VG150 trong repo này.

SegPVSG release được kiểm tra vẫn có logic temporal support tương tự trong [`utils/eval_metrics.py`][seg-metrics]. Đây là nhận xét về source đã đọc; chưa chạy upstream evaluator với dataset hoặc tái kiểm nghiệm điểm paper.

### Protocol chính đề xuất: `pvsg-gt-trd-tiouv1`

| Thuộc tính | Quy định |
|---|---|
| Object input | GT masks + class labels + track IDs; lưu rõ ba cờ oracle |
| Candidates | Tất cả ordered track pairs `s != o` có đồng hiện; không chỉ các pair có GT relation |
| Unit đánh giá | Một `(video,s,p,o)` với hợp tất cả interval, không đếm từng lần xuất hiện thành GT riêng |
| Class prediction | Multi-label; nhiều predicates/cặp được phép xuất hiện trong top-K |
| Ranking | Một score cho mỗi tuple, top-K **toàn video**, không phải top-K/frame/cặp/cửa sổ |
| Matching | Đúng track IDs có thứ tự và predicate; `tIoU >= τ` |
| Temporal support | Đánh giá trên lưới frame gốc 5 FPS; GT giao visibility của cả hai tracks. Prediction chỉ mask bằng visibility, không được cắt theo GT relation span |
| Thresholds | Chính `τ=0.5`; phụ `0.3`, `0.7` để kiểm tra chất lượng boundary. Đây là tIoU thresholds của GML |
| R@K | `Σ_video hits@K / Σ_video số GT tuples`, K = 20, 50, 100 |
| mR@K | Tính recall gộp riêng từng predicate rồi lấy trung bình các lớp có support; xuất support và số lớp tham gia |
| Duplicate/empty | Gộp tuple trùng; stable tie-break; không cho duplicate hit nhiều lần; empty GT/split được báo riêng |
| Selection | Chọn checkpoint bằng internal-val mR@50 tại tIoU=0,5; tie-break R@50 rồi epoch sớm hơn |

GT relation không còn frame hợp lệ sau visibility filtering phải được thống kê riêng. Nếu gặp trường hợp này, dừng để chốt rule trong manifest trước khi train, không âm thầm lọc khác nhau theo model. Chỉ publish điểm chính sau khi audit coverage; có thể thêm raw-annotation diagnostic với tên khác.

Metric phụ: predicate-only R/mR khi bỏ điều kiện thời gian; temporal IoU trên **tất cả GT tuples, gán 0 cho tuple bỏ sót**; F1 theo frame và segment-F1 có one-to-one matching. Segment-F1 dùng đơn vị interval, nên là diagnostic riêng, không trộn với tuple recall. Báo cáo thêm head/mid/tail theo tần suất **train**, domain nguồn, thời lượng quan hệ, số parameters, VRAM và thời gian chạy.

Giữ một evaluator compatibility riêng nếu cần nối với OpenPVSG. Phải pin commit, tái hiện chính xác code upstream và ghi metric namespace riêng; không âm thầm “sửa metric” rồi so số mới với bảng paper. Bản GT-track compatibility vẫn là oracle-input result. `rel_test.py` và các hàm NumPy là nền tham khảo gọn hơn việc kéo nguyên MMDetection vào trainer.

## 6. Thiết kế thực nghiệm GML

### Đặc trưng và supervision dùng chung

**Bản tối thiểu:** class embedding 64 chiều + geometry 6 chiều `(cx,cy,w,h,box_area,mask_area)` + motion `(Δcx,Δcy,Δw,Δh)` tính theo bước thời gian + cờ hiện diện/gap. Chuẩn hóa bằng kích thước frame; motion chỉ tính giữa hai quan sát hợp lệ, không lấy hiệu với padding. Thiếu thông tin RGB làm quan hệ hành động khó hơn; dùng bản này để kiểm tra data/evaluator, không kết luận negative về GNN từ riêng bản geometry.

**Bản thực nghiệm chính:** thêm vector vùng ảnh 512 chiều từ CLIP ViT-B/32 đóng băng, cùng crop/preprocessing/cache cho mọi model. Đây là lựa chọn thiết kế, không phải feature đúng nguyên bản của các paper. Masked crop là bản đầu dễ triển khai; crop box giữ context hoặc masked dense pooling là ablation sau. Không fine-tune CLIP/segmentation.

Các pixel ngoài mask được điền màu nền cố định, crop theo box và resize bằng cùng quy tắc. Thêm pair geometry cho mọi model: vị trí tương đối, log tỉ lệ kích thước, box IoU, chênh lệch vận tốc. Không đưa GT relation flags, GT interval, relation caption hoặc thông tin ontology từ test vào features.

Targets `y[s,o,t,p]∈{0,1}` được rasterize từ interval GT; nhiều predicate có thể bằng 1. Frame không đồng hiện hoặc padding có loss mask 0. Các cặp không gán relation được coi là âm theo annotation benchmark; phải ghi hạn chế annotation thiếu, không gọi đó là bảo đảm “không có quan hệ trong thế giới thực”.

### Graph construction

- Node là `(track_id, frame_id)` ở frame object hiện diện.
- Cạnh **spatial**: mọi cặp object khác nhau trong cùng frame, hai chiều. GAT tự thêm self-loop. Đây là topology từ input objects, không từ GT predicates.
- Cạnh **temporal**: nối cùng track ở `t↔t+1`; khi có khoảng mất quan sát, mặc định không nối qua gap. Ablation có thể nối tối đa gap đã khóa và thêm `Δt`.
- Dùng module/weight riêng cho spatial và temporal message passing; thêm temporal position hoặc `Δt` để mô hình biết thứ tự, thay vì chỉ một graph chain vô hướng không có thời gian.
- Cạnh query để chấm predicate là ordered pair trong frame, độc lập với cạnh message passing. Xóa cạnh GNN trong ablation không được xóa các cặp cần dự đoán.
- Không dựng attention dense cho tất cả `(N×T)^2` node pairs. Complexity xấp xỉ `O(TN² + TN)` cho graph edges; pair head vẫn có chi phí `O(TN²C)`.

```mermaid
flowchart LR
    A[RGB và GT tracks] --> B[Features đóng băng và geometry]
    B --> C[MLP hoặc Static GAT hoặc ST-GAT]
    C --> D[57 predicate logits mỗi cặp mỗi frame]
    D --> E[Ghép cửa sổ và giải mã nhiều intervals]
    E --> F[Top-K tuples theo video]
    F --> G[GT-track tIoU evaluator]
```

### Các model bắt buộc

| Model | Encoder | Relation head | Câu hỏi thực nghiệm |
|---|---|---|---|
| **MLP** | Projection từng object, không message passing, không temporal aggregation học được | MLP trên `[h_s,h_o,h_s*h_o,h_s-h_o,pair_geometry]` → 57 logits/frame | Reference phi đồ thị |
| **Static GAT** | Hai spatial GAT layers từng frame; hidden 128, 4 heads, residual/norm | Cùng pair head và decoder với MLP | Context giữa các object trong frame có ích không? |
| **ST-GAT** | Spatial GAT xen temporal GAT dọc cùng track, hai blocks; hidden 128, 4 heads; có temporal position | Cùng pair head/decoder | Thêm temporal message passing có cải thiện nhận dạng và interval không? |

Để tách hiệu quả temporal context khỏi hiệu quả graph, thêm đối chứng rẻ **MLP + temporal Conv1D** trên từng pair sequence, kernel 3, receptive field gần ST-GAT. Đây là baseline bổ trợ quan trọng nếu muốn nói lợi ích đến từ GNN, không chỉ do nhìn nhiều frame hơn. Các kích thước trên là cấu hình khởi đầu; chưa có đo GPU/accuracy.

Hai temporal layers chỉ nối frame liền kề có receptive field tối đa ±2 frame; window 64 không tự làm mọi node nhìn đủ 64 frame. Nếu cần context dài, ablate temporal edges có độ trễ `{1,2,4}` hoặc thêm layers, đồng thời tăng receptive field của Conv1D control tương ứng. Ghi số parameters vì ST-GAT có thêm layers so với Static GAT; dùng ablation bỏ cạnh trên cùng kiến trúc để kiểm tra ảnh hưởng topology.

### Loss functions

Loss bắt buộc là **masked BCEWithLogits** trên 57 channels. Không thêm background vào softmax; no-relation được biểu diễn bằng vector toàn 0.

```text
L_frame = sum(m * BCEWithLogits(z, y; pos_weight)) / số phần tử hợp lệ
L = L_frame + λ_boundary * L_boundary + λ_smooth * L_smooth
```

Khởi đầu `pos_weight=1`, `λ_boundary=λ_smooth=0` để baseline rõ ràng. Bản xử lý long-tail dùng `pos_weight[p] = clip(sqrt(n_neg[p]/max(n_pos[p],1)), 1, 10)` từ **train**, áp dụng giống nhau cho mọi model; so sánh riêng với unweighted BCE. Nếu sampling âm làm thay đổi phân bố, thống kê/reweight phải nhất quán với train sampler và ghi trong config. Không đồng thời thêm focal loss, class weighting và oversampling ở lần đầu.

Khi cần boundary chính xác hơn, head phụ dự đoán start/end multi-label cho từng predicate, targets từ chuyển trạng thái 0→1 và 1→0; dùng masked BCE. Không đánh dấu biên giả tại điểm cắt cửa sổ: dùng nhãn full-video để dựng targets, ignore biên bị cắt cụt. `L_smooth` tùy chọn phạt biến thiên xác suất giữa frame liền kề có cùng trạng thái GT và visibility; không làm trơn qua boundary GT. Mỗi thành phần phải có ablation riêng.

Một head hồi quy duy nhất `(start,end)` cho mỗi pair không biểu diễn được quan hệ tái diễn. Proposal/DETR interval regression là bước nâng cao nếu framewise baseline thất bại, không nằm trong MVP.

### Dự đoán cả predicate và temporal interval

1. Model xuất 57 xác suất sigmoid/frame. Nếu chia cửa sổ, ghép logits về timeline video bằng trung bình phần overlap **trước** threshold.
2. Mask những frame không có cả subject và object. Với mỗi predicate, threshold `θ=0.5` mặc định; dùng **một θ chung** chọn từ `{0.3,0.5,0.7}` trên internal-val nếu cần.
3. Run-length encoding tạo mọi đoạn liên tiếp active; giữ nhiều đoạn rời nhau cho cùng tuple. Mặc định không gộp gap, không xóa đoạn ngắn. Có thể ablate `gap≤1 frame`, `min_length∈{1,2,3}`, chỉ tune trên internal-val và áp dụng cùng policy cho tất cả models.
4. Score tuple = trung bình xác suất trên các frame active dự đoán; không active thì không xuất. Các đoạn cùng tuple được hợp lại, rồi rank toàn video. Score dùng chung này có thể ưu tiên spike ngắn, nên so với masked temporal max/top-q mean trong một ablation; không chọn theo test.
5. Đổi `[start,end]` inclusive thành thời gian `start/5` đến `(end+1)/5` giây theo quy ước frame bins, lưu cả frame indices và FPS. Không suy ra FPS từ FPS video gốc.

Để scale, bắt đầu window 64 frame, stride 32 ở 5 FPS (~12,8 giây/cửa sổ). Train sample windows theo video; validation/test chạy phủ **toàn bộ** timeline, gồm cửa sổ cuối. Không resize mọi video về cùng T hoặc cắt bỏ sau frame 900. Test top-K chỉ thực hiện sau ghép toàn video.

### Đối chứng và tái lập CVPR 2023

Ưu tiên **baseline 1D Convolution** trong PVSG CVPR 2023: đơn giản, học temporal context và phù hợp làm đối chứng cho ST-GAT. Transformer là lựa chọn tiếp theo nếu còn ngân sách; không yêu cầu chạy đủ cả bốn relation heads hoặc tái huấn luyện segmentation. [Các baseline trong paper][pvsg-paper].

Phân biệt rõ hai mức triển khai:

| Mức | Cách thực hiện | Cách gọi kết quả |
|---|---|---|
| **A — Baseline chuyển thể cho GML**, ưu tiên | Dùng khối temporal Conv của PVSG trên cùng GT-track feature cache; giữ output 57 logits/frame, BCE và decoder/evaluator chung với ba model GML | `pvsg-conv-gt-adapt-v1`: chuyển thể baseline CVPR 2023 sang GT-track TRD; so sánh trực tiếp với GML, không gọi là tái lập điểm paper |
| **B — Tái lập benchmark gốc**, nếu đủ artifacts | Dùng checkpoint segmentation/tracking đã công bố ở chế độ suy luận; giữ query features, object encoders, pair proposal, relation heads/losses và evaluator theo recipe upstream; khóa đúng annotation/split/checkpoint | `openpvsg-ips-conv-repro-v1`: nhánh predicted-track riêng; báo điểm, chênh lệch và mọi sai khác so với cấu hình công bố |

Nếu so trực tiếp GAT với Conv trên predicted tracks, chạy IPS+Tracking một lần và lưu kết quả. Cả hai relation head dùng cùng bản track, query features, danh sách cặp ứng viên, tập video và evaluator OpenPVSG. Báo kết quả trong bảng predicted-track. Chỉ tính chênh lệch với số công bố khi đã đối chiếu đúng split, annotation và protocol.

Ở mức B, có thể đánh giá checkpoint relation có sẵn trước, rồi tái huấn luyện relation module nếu thực hiện được. Chỉ chạy checkpoint và evaluator là **tái đánh giá checkpoint**, không đồng nghĩa tái lập quá trình huấn luyện. Nếu thiếu checkpoint, split gốc hoặc recipe, ghi giới hạn cụ thể; không chuyển sang tự train detector/backbone để hoàn thành mục này.

**Sai khác paper/code cần xử lý trước khi cấu hình:** §5.2 bản **CVF** mô tả Conv kernel 5 với 3 layers và Transformer 3 layers; bản PDF cập nhật trên trang tác giả chỉ nêu kernel 5 cho Conv, nhưng constructor ở snapshot đã kiểm tra mặc định `Learnable1DConv(..., num_layers=1)` và `TemporalTransformer(..., num_transformer_layers=1)`. Vì vậy truyền số layers tường minh, ghi rõ `paper-spec` hay `released-code`; không dựa vào default rồi gọi là đúng paper. Code release còn có `span_head` theo frame và `pred_head` với temporal max pooling. Mức A dùng head/decoder chung của GML nên phải ghi phần chuyển thể này. [Conv source][openpvsg-conv], [Transformer source][openpvsg-transformer].

MLP+Conv1D kernel 3 ở phần ablation là control khớp receptive field; **không mặc nhiên là baseline Conv CVPR 2023**. Nếu triển khai cả hai, ghi cấu hình và mục tiêu riêng. Để giữ khối lượng vừa sức, ưu tiên pilot mức A; giữ các ablation bỏ spatial/temporal edges dưới dạng tùy chọn chạy sau, không tự xếp vào queue. Mức B là mục tiêu bổ sung khi artifacts đủ, không đặt điều kiện hoàn thành MVP vào việc đạt lại đúng số paper.

### Ma trận ablation tối thiểu

| So sánh | Giữ cố định | Diễn giải được |
|---|---|---|
| MLP / Static GAT / ST-GAT | Input, split, pair head, loss, decoder | Spatial context và temporal context |
| ST-GAT bỏ temporal edges | Spatial graph, query pairs, số bước tối ưu | Lợi ích đường truyền theo track |
| Static/ST-GAT bỏ spatial edges | Self-loop, temporal edges nếu có, query pairs | Lợi ích trao đổi giữa object |
| Temporal track edges nối sai có kiểm soát | Số cạnh, frame và degree gần tương đương | Temporal identity có quan trọng không? |
| MLP + Conv1D / ST-GAT | Receptive field và parameters gần tương đương | Temporal sequence model so với graph model |
| Không/có visual features | Geometry, semantics, protocol | Mô hình có chỉ dựa vào nhãn/prior không? |
| BCE / weighted BCE; decoder thô / lọc | Encoder và validation budget | Mức tăng do model hay loss/hậu xử lý |

Spatial graph đầy đủ chỉ đổi thứ tự edges sẽ không đổi topology. Vì vậy không gọi thao tác permute danh sách cạnh là “shuffle graph” có ý nghĩa; dùng edge removal/dropout hoặc graph thưa rồi degree-preserving rewiring. Không shuffle thứ tự frame chỉ ở test rồi coi đó là ablation thuần: đó còn là distribution shift.

Bảng chính so sánh MLP, Static GAT, ST-GAT và Temporal Conv chuyển thể trên cùng GT tracks, đặc trưng CLIP và evaluator M1. Các ablation bỏ cạnh spatial/temporal vẫn có thể chạy khi cần phân tích đóng góp của từng loại cạnh.

## 7. Protocol huấn luyện và tái lập

1. **Đã khóa 338 train / 62 test IDs** từ bản phát hành của tác giả ở M0; giữ nguyên 62 test video cho đánh giá cuối. Tách internal-val khoảng 15% chỉ trong 338 train, phân tầng theo nguồn; nếu nhiều clip cùng source video/session thì group chung. Không chia ngẫu nhiên theo frame hoặc window.
2. **Đã khóa 287 fit / 51 internal-val / 62 test** ở [manifest M2](../../experiments/pvsg_trd_v1/m2/manifests/dataset.lock.json), seed 17; 62 test IDs giữ nguyên. Fit/internal-val không chung source group suy ra từ ID. Official train/test có 4 Ego4D source UUID groups chung; ghi giới hạn này, không tự sửa outer split, chưa kết luận có frame trùng. Đây là phần tách phục vụ chọn hyperparameters bên trong official train, không thay đổi outer benchmark split. Khi train trên fit subset, báo rõ số video thực dùng và không mô tả là tái lập đủ train budget của bản phát hành.
3. Mọi scaler, class weights, priors, tail bins và lựa chọn hyperparameter lấy từ fit/internal-val đúng vai trò. Không tune trên `val` upstream đang đóng vai trò test. Nếu không có lớp hiếm trong internal-val, báo support thay vì tune ngưỡng riêng lớp đó.
4. Chính: 3 models × seeds `{17,42,2026}` = **9 runs**, chạy tuần tự. Pilot/ablations chọn trước và ghi riêng. Conv CVPR 2023 mức A chạy pilot seed 17. Hoàn tất cùng bộ ba seeds cho bảng chính, tổng cộng **12 runs** gồm 9 M3 và 3 Conv M4 (tính pilot nếu giữ nguyên cấu hình). MLP+Conv1D control và nhánh tái lập mức B có kế hoạch chạy riêng; không tự thêm vào queue. Không có run KG.
5. Recipe M3 đã triển khai: AdamW LR `1e-3`, weight decay `1e-4`, dropout `0.1`, tối đa 50 epochs, tối thiểu 10 trước early stopping, patience 10 theo metric chính; LR plateau theo validation BCE. Theo yêu cầu tận dụng GPU, physical batch sẽ đo riêng từng model trong 1–32 windows, giữ **effective batch 32** bằng accumulation. Mỗi epoch sample 4 windows/video fit = 1.148 windows (36 optimizer steps), internal-val phủ đủ 545 windows. Đây là thay đổi budget so với dự kiến microbatch 1/accumulation 4 ban đầu, áp dụng cùng cho cả ba model và ghi trong config. Chunk pair head, kiểm peak reserved bằng envelope 38 object/frame; chưa có số đo GPU và không cắt bỏ tracks.
6. Train lần đầu xét mọi cặp trong window. Nếu quá chậm, sampling âm chỉ ở train và phải giữ đầy đủ positives, ghi tỷ lệ/probability correction; inference chấm mọi cặp đồng hiện theo chunk, không dùng GT để chọn pair.
7. Mean ± std qua ba seeds; không chỉ báo best seed. So sánh paired trên cùng video; bootstrap theo video có thể bổ sung CI. Không hứa GAT chắc chắn thắng MLP hoặc temporal Conv baseline. Nhánh tái lập chỉ có một seed phải ghi rõ, không gộp với bảng ba-seed.
8. WandB bắt buộc cho các lệnh train chính: project/group riêng `pvsg-trd-v1`, config, seed, split/features/code hashes, loss, R/mR, tIoU, epoch, VRAM, runtime và checkpoint hash. Launcher phải kiểm tra logging hoạt động trước train và lưu run ID để resume.

## 8. Tài nguyên, dependencies và khó khăn

### Dung lượng và máy hiện có

Metadata kho chính thức Hugging Face ghi **10,8 GB** tổng archive và `pvsg.json` khoảng **3,88 MB**. Các file nén: Ego4D masks 940 MB + videos 3,8 GB; EPIC-KITCHENS 742 MB + 3,53 GB; VidOR 353 MB + 1,42 GB. Đây là kích thước hiển thị, làm tròn; không phải dung lượng sau giải nén. M2 đã tải và kiểm SHA256 cả 6 archive: **10.792.485.748 bytes**. Chỉ giải nén MP4, đọc masks trực tiếp từ ZIP; chưa ghi RGB frames rời. [Kho HF][hf-root], [Ego4D][hf-ego], [EPIC-KITCHENS][hf-epic], [VidOR][hf-vidor].

Audit M2 giải mã CPU đủ **400 video / 149.488 masks**. Header frame count khác masks ở 7 video; 5 trường hợp giải mã thực tế đủ frame. Còn 4 RGB frames thiếu: một frame cuối VidOR, ba frame trong gap PTS của một Ego4D clip. Ghép RGB theo timestamp 5 FPS, giữ geometry và cờ visual availability tại frame thiếu. Bốn video có mask ID không có object label; vùng này được loại khỏi inputs và ghi ledger. [Media audit](../../experiments/pvsg_trd_v1/m2/provenance/media_audit.json).

Read-only system inspection trong phiên nghiên cứu: **RTX 5090, 32607 MiB VRAM (~31,8 GiB)**; RAM tổng 60 GiB; filesystem dự án còn khoảng **171 GiB**, đã dùng 91%. Không chạy CUDA workload hoặc cài package để lấy thông tin này.

Ước lượng thiết kế, **chưa đo**:

- Cache 512-d fp16 cho 150K frames và trung bình 8–20 object hiện diện/frame: `150000 × (8..20) × 512 × 2` ≈ **1,23–3,07 GB**, chưa gồm index/geometry/metadata. Con số object/frame chỉ là giả định lập ngân sách.
- Không cache feature của mọi pair: dung lượng tăng theo N². Ghép pair features lúc train/inference, chunk pair head.
- 150K ảnh JPEG trung bình 200 KB đã khoảng 30 GB; masks, archives, môi trường và outputs làm tổng lớn hơn. Dự trù **80–120 GB** cho một bản làm việc và cache, kiểm tra số thực tế sau sample extraction; chưa bảo đảm toàn bộ giải nén vừa ngân sách này.
- Hướng train trên feature cache dự kiến phù hợp GPU 12–24 GB và GPU hiện có; hidden 128 + window 64 chỉ là cấu hình xuất phát. Chi phí crop feature extraction có thể lớn hơn training relation head. Chưa thể báo thời gian train/FPS khi chưa biết số visible nodes/pairs và throughput.

### Môi trường

| Hướng | Phụ thuộc | Nhận định |
|---|---|---|
| GML GT-track mới | M2: PyTorch 2.9.1+cu128, CLIP visual vendored/pinned, NumPy 2.1.0, Pillow 11.2.1, PyAV 16.0.1, SciPy 1.15.3; M3 thêm GNN/WandB | Overlay environment riêng, không dùng torchvision đang không tương thích hoặc MMDetection; clean requirements trong `m2/` |
| OpenPVSG nguyên bản | README hướng CUDA 10.1, `mmcv-full==1.4.0`, `mmdet==2.25.0`, tracker extensions | Có rủi ro build/ABI lớn khi đưa lên GPU và stack mới |
| SegPVSG release | Python 3.9, torch 1.10.2, mmcv-full 1.6.0, mmdet 2.25.0; README môi trường RTX 3090 | Không suy ra chạy trực tiếp được trên RTX 5090 |

Nguồn: [OpenPVSG environment hướng dẫn][openpvsg-readme], [SegPVSG README][seg-code]. Repo hiện có [`requirements.txt`](../../requirements.txt) dùng torch 2.7.0/PyG 2.6.1 và [`requirements-level3-sgg-robin-cu128-v2.txt`](../../requirements-level3-sgg-robin-cu128-v2.txt) tách runtime torch 2.7.1+cu128 cho RTX 5090. Có thể tham khảo cách tách môi trường này; **không cài đè hoặc lấy cả dependencies Level 3 cho TRD**. Cần khóa requirements riêng sau CPU checks và kiểm tra tương thích CUDA/PyG ở milestone triển khai.

### Rủi ro triển khai và cách giảm

| Khó khăn | Tác động | Cách xử lý trong kế hoạch |
|---|---|---|
| Paper, code và split khác version | So sánh sai benchmark | Pin annotation/evaluator/source; có bảng protocol riêng |
| Nhiều đoạn, nhiều nhãn, mất quan sát | Sai temporal targets hoặc loss | Binary multi-label timeline, visibility masks, inclusive endpoints; fixture CPU |
| Track ID/class ID/frame offset khác nhau | Nhãn đúng nhưng ghép sai object | Bijection ID map và kiểm tra overlay sample |
| Background/stuff có mask lớn; occlusion và camera motion | Geometry/motion đơn giản chưa đủ | Thêm visual features; phân tích theo nguồn/loại quan hệ, tránh suy luận camera motion = object action |
| Long-tail, ít video, negative pairs lớn | MLP học prior, mR bất ổn | Support theo lớp, ba seeds, BCE control, head/tail analysis |
| Official query features khác CLIP crops | Không tái lập trực tiếp paper baseline | Gọi rõ “GML baseline trên GT tracks”; cùng features giữa models |
| Pipeline relation set loại unmatched tracks | Recall bị tính trên tập dễ hơn | GT evaluator đọc raw annotation, không thừa hưởng filtered relation set |
| Source download cũ | Không lấy được dữ liệu/checkpoints | Kho HF tác giả là lựa chọn; kiểm tra hash/revision khi triển khai. SharePoint được thử đọc trong khảo sát chuyển sang đăng nhập |
| Dense tracks/windows lớn | VRAM/time tăng mạnh | Node/edge budget, pair chunking, overlap windows; tuyệt đối không bỏ video test khó |
| Paper-spec khác default code baseline | Tái lập sai kiến trúc mà không nhận ra | Khóa số layers, heads, loss, features và decoder; ghi `paper-spec` hoặc `released-code` |

Không cần tải toàn bộ VidOR/Ego4D/EPIC-KITCHENS gốc cho MVP nếu bản PVSG subset đủ RGB/masks/annotations. Trước phát hành lại dữ liệu phải đọc điều kiện của bản tải và nguồn video; nhãn license ở trang mirror không tự giải quyết quyền của mọi video nguồn.

## 9. Tổ chức implementation dự kiến

M0 đã tạo vùng riêng với `protocol.json`, schemas, manifest split, provenance, synthetic examples và validator CPU. M1 đã triển khai `data.py`, `contracts.py`, `intervals.py`, `graphs.py`, `decoding.py`, `evaluation.py`, `cli.py` và synthetic fixtures/tests. M2 đã có `m2/` riêng chứa download/media/audit/target adaptation/internal split/feature extractor/provenance và launcher. Model, loss và training runner vẫn là kế hoạch M3; cây dưới đây thể hiện thiết kế tổng thể, không phải tất cả file đã tồn tại:

```text
experiments/pvsg_trd_v1/
  README.md
  protocol.json           # đặc tả M0 đã khóa
  validate_m0.py           # kiểm tra cấu trúc, split và provenance trên CPU
  schemas/                # hợp đồng input/targets/predictions/manifest
  manifests/              # 338/62 đã khóa; template 360/40 chỉ để truy vết
  examples/               # synthetic contracts M0
  data.py                 # parser, ID mapping, temporal targets
  features.py             # frozen encoder/cache contract
  graphs.py               # spatial/temporal topology
  models.py               # MLP, Static GAT, ST-GAT
  pvsg_baselines.py       # Conv CVPR 2023 chuyển thể; Transformer nếu cần
  losses.py
  decoding.py
  evaluation.py           # pvsg-gt-trd-tiouv1
  upstream_compat.py      # chỉ nếu cần benchmark compatibility
  runner.py               # CPU preflight, train/evaluate entrypoints
  configs/
  tests/                  # fixtures nhỏ, không phụ thuộc dataset đầy đủ
  provenance/             # lock và source references
requirements-pvsg-trd-v1.txt
artifacts/pvsg-trd-v1/     # features/manifests, ngoài Git
reports/pvsg-trd-v1/       # metrics/case studies theo run
```

`provenance` ghi commit parent, code import hoặc chuyển thể, upstream commit/license, annotation SHA256, split hash, class mapping hash, feature model/weights/preprocessing hash, decoder/evaluator version và train config. Cache GT và predicted tracks dùng namespace khác. Không đưa artifacts lớn hoặc những thay đổi cá nhân đang có vào commit công việc mới.

## 10. Roadmap và điều kiện hoàn thành

Thời lượng là ước lượng công việc triển khai, chưa gồm thời gian chờ tải/chạy do người dùng thực hiện.

| Milestone | Công việc | Điều kiện hoàn thành | Ước lượng |
|---|---|---|---|
| **M0 — Hoàn thành: khóa phạm vi và protocol** | Xác minh bản phát hành tác giả năm 2023; người dùng chọn 338/62; khóa ID/hash, GT-track inputs/metrics, schema và manifest | [Artifacts M0](../../experiments/pvsg_trd_v1/README.md); protocol riêng; ghi sai khác CVF 360/40; kiểm tra CPU; không tự dựng split | Đã thực hiện |
| **M1 — Hoàn thành: parser/evaluator CPU** | Adapter tách input/targets, multi-label timelines, decoder, evaluator, graph batching; đối chiếu Python-set reference | [97 tests CPU đạt và bàn giao M1](../../experiments/pvsg_trd_v1/README_M1.md); các ca perfect/wrong/overlong/K/duplicate/visibility/batching đều đạt; audit raw train ghi nhận 34 video có lỗi cần M2 | Đã thực hiện |
| **M2 — CPU/data hoàn tất; chờ full visual cache GPU** | 6 archives + checkpoint đã kiểm hash; audit toàn bộ media; policy nhãn có ledger; split 287/51/62; overlay 19 video; extractor/resume và CPU CLIP probe | [Bàn giao M2](../../experiments/pvsg_trd_v1/m2/README.md); còn chạy launcher GPU và xác nhận cache 400 video/storage/throughput. Giữ giới hạn 4 outer source groups chung | Người dùng chạy extraction; chưa đo thời gian GPU |
| **M3 — Code/CPU sẵn sàng; chờ GPU probe và train** | MLP/Static GAT/ST-GAT; sparse windows/decoder; chunked gradient; sizing probe; online WandB; resume và queue 9 runs | [README M3](../../experiments/pvsg_trd_v1/m3/README.md). Physical batch chưa chốt: đo 1–32 windows sau M2, effective batch 32; tối đa 50 epochs/patience 10. Có CPU overfit/real-window checks; chưa có GPU/convergence benchmark | CPU/code đã chuẩn bị; GPU/train chưa chạy |
| **M4 — Code/CPU sẵn sàng; chờ GPU và kết quả** | Ưu tiên Conv k5 × 3 theo CVF 2023 chuyển thể trên GT tracks; giữ `st_no_spatial`, `st_no_temporal`, `st_no_edges` và Conv k3 × 2 dưới dạng tùy chọn | [README M4](../../experiments/pvsg_trd_v1/m4/README.md). Module/provenance riêng, reuse loss/evaluator M3; profile riêng từng model; một lệnh một run, WandB online. Chưa phải tái lập end-to-end; không KG | CPU/code đã chuẩn bị; GPU/train chưa chạy |
| **M5 — Đánh giá đóng băng và báo cáo** | Freeze config rồi evaluate held-out một lượt kế hoạch; per-class/domain/tail; 5 đúng + 5 sai và timeline visualization | Mean±std; bảng protocol; giới hạn GT oracle; artifact tái lập; liên hệ với mục tiêu môn học | 1–2 ngày |

**M4 đã chuẩn bị riêng trong `experiments/pvsg_trd_v1/m4/`:** ưu tiên `pvsg_conv_k5_l3` (437.945 params, temporal RF 13), control tùy chọn `mlp_conv_k3_l2` (290.361 params, RF 5) và ba ablation ST-GAT cùng 226.873 params. Bỏ message edges vẫn giữ self-loops, toàn bộ query pairs và loss/decoder. Conv chia sequence theo window/directed pair/đoạn co-visible liên tục, không nối qua gap hoặc trộn video. Model dùng CLIP/GT tracks, M3 pair projection/head, masked BCE và M1 tIoU evaluator; ghi rõ **GT-track adaptation**, không so trực tiếp số end-to-end paper. Physical BS sẽ đo riêng 1–32, effective BS 32, tối đa 50 epochs/patience 10 như M3. CPU fixture overfit đạt positives ở cả năm model; chưa có kết luận hội tụ trên fit/validation thật. Lệnh profile và train một model/seed, không tự chạy queue, nằm trong README M4. Chỉ user khởi chạy train chính, WandB online bắt buộc.

CPU fixtures M1 đã kiểm tra: nhiều interval rời; nhiều predicates cùng pair; subject/object đảo chiều; `end` inclusive; clip boundary; mask rỗng; pair chưa từng có relation; lớp không có support; whole-video negative; ghép overlap không làm nhãn/score đổi do window order. Có đối chiếu evaluator với Python sets độc lập trên 12 seeds dữ liệu giả, kiểm tra cohort đầy đủ và CLI. Đây là kiểm tra semantics, không phải kết quả inference trên PVSG thật.

Trước tác vụ GPU ở giai đoạn triển khai: hoàn tất code/CPU checks, commit phần liên quan và bàn giao lệnh có phạm vi rõ. Train chính do người dùng chạy, WandB bắt buộc; launcher ghi **model, microbatch/effective batch, seed, output, resume, số run và có tự chạy tiếp queue hay không**. Mặc định một lệnh một run, không tự chuyển model; nếu bàn giao queue thì phải nêu đủ thứ tự, chạy tuần tự một model mỗi thời điểm. M0 không tải media/checkpoint, chạy GPU hoặc training.

**MVP đủ giá trị cho phần mở rộng GML:** ba models dùng cùng GT-track feature cache, evaluator tIoU đúng, ba seeds và phân tích lỗi; không có KG. Theo ưu tiên mới, chuẩn bị Conv CVPR 2023 chuyển thể làm đối chứng trước; giữ code ablation cạnh thời gian/spatial để chạy khi cần, không tự đưa vào queue. Nhánh tái lập gốc phụ thuộc artifacts. Không cần triển khai full UNO/MCL/SegPVSG để đạt mục tiêu mở rộng này. Đây là đánh giá phạm vi thực nghiệm, không khẳng định TRD tự đáp ứng toàn bộ tiêu chí chỉnh sửa/tính nhất quán video hoặc bảo đảm điểm của đồ án.

## 11. Nguồn và snapshot

Các repo được đọc qua GitHub tại snapshot sau; PDF chỉ dùng để nghiên cứu. Dataset size lấy từ metadata trang HF, không phải phép đo sau download.

| Nguồn code | Commit đã kiểm tra |
|---|---|
| OpenPVSG | `b2850b980c272368215a754299986cb904d55820` |
| MCL | `41f25ae06157f3db06cd4d023107b6d692435dd9` |
| UNO | `6c05c555c01cce3f6bb2d2da179e4bee7e7877b6` |
| SegPVSG | `12e7d1f38c4fe30ae717a4173dfee121154411fc` |

Đã xác minh M0/M1: annotation identity/split/vocabulary, parser/evaluator và tests CPU. M2 đã tải đủ media và CLIP checkpoint, audit timeline/masks/visibility, review overlay 19 video, khóa internal split, version target policy và kiểm CLIP thật trên CPU. Chưa có full visual cache, throughput/VRAM GPU, CUDA/PyG training runtime, batch size tối đa, training time hoặc replication numbers. CLIP weights đã tải không có nghĩa checkpoint IPS/relation upstream để tái lập gốc đã đủ. Các bước này vẫn cần thực hiện ở lần chạy GPU/M3/M4.

[pvsg-page]: https://jingkangyang.com/PVSG/
[pvsg-paper]: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Panoptic_Video_Scene_Graph_Generation_CVPR_2023_paper.pdf
[pvsg-author-paper]: https://jingkangyang.com/PVSG/static/pdfs/pvsg.pdf
[hf-annotation]: https://huggingface.co/datasets/Jingkang/PVSG/resolve/7e5f1ec9fd8f323182e84d990819854bb72da478/pvsg.json
[openpvsg-readme]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/README.md
[openpvsg-dataset]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/datasets/datasets/pvsg_relation.py
[openpvsg-matching]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/utils/relation_matching.py
[openpvsg-metrics]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/utils/rel_metrics.py
[openpvsg-rel-test]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/tools/rel_test.py
[openpvsg-full-test]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/tools/rel_test_full.py
[openpvsg-conv]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/models/relation_head/convolution.py
[openpvsg-transformer]: https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/models/relation_head/transformer.py
[mcl-paper]: https://arxiv.org/pdf/2412.07160v3
[mcl-code]: https://github.com/nguyentthong/motion-contrastive-sgg/blob/41f25ae06157f3db06cd4d023107b6d692435dd9/README.md
[uno-paper]: https://openaccess.thecvf.com/content/WACV2026/papers/Le_UNO_Unifying_One-stage_Video_Scene_Graph_Generation_via_Object-Centric_Visual_WACV_2026_paper.pdf
[uno-code]: https://github.com/Fsoft-AIC/UNO/blob/6c05c555c01cce3f6bb2d2da179e4bee7e7877b6/README.md
[seg-venue]: https://proceedings.mlr.press/v306/li26bx.html
[seg-paper]: https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26bx/li26bx.pdf
[seg-code]: https://github.com/ticatt/SegPVSG/blob/12e7d1f38c4fe30ae717a4173dfee121154411fc/README.md
[seg-metrics]: https://github.com/ticatt/SegPVSG/blob/12e7d1f38c4fe30ae717a4173dfee121154411fc/utils/eval_metrics.py
[hf-root]: https://huggingface.co/datasets/Jingkang/PVSG/tree/main
[hf-ego]: https://huggingface.co/datasets/Jingkang/PVSG/tree/main/Ego4D
[hf-epic]: https://huggingface.co/datasets/Jingkang/PVSG/tree/main/EpicKitchen
[hf-vidor]: https://huggingface.co/datasets/Jingkang/PVSG/tree/main/VidOR

### Kiểm tra hội tụ muộn sau M3/M4 (phân tích bổ sung)

Sau khi xem 24 kết quả validation, chọn bảy lượt đạt R@50 và mR@50 bằng 0, dừng ở epoch 16 (epoch 1 là best, tiếp theo 15 epoch không cải thiện). ST-GAT và ST-GAT bỏ cạnh thời gian mỗi model dùng seed 17, 2026. ST-GAT chỉ giữ self-loop dùng seed 17, 42, 2026.

Chạy lại từ epoch 1 với cùng seed, batch 32, optimizer, scheduler và tiêu chí chọn checkpoint. Thay đổi duy nhất trong lịch học là min_epochs từ 10 lên 50. Patience 15 vẫn tích lũy trong suốt quá trình train. Epoch 50 là lần đầu có thể dừng; nếu metric tiếp tục cải thiện thì được train đến tối đa 150. Do phép toán GPU có thể không tất định, ghi cả điểm tốt nhất trong 16 epoch đầu của lần chạy lại.

Báo từng seed trong phụ lục “Kiểm tra hội tụ muộn theo yêu cầu sau khi xem validation”, gồm điểm gốc, điểm ở lần chạy bổ sung, epoch đầu tiên R@50 dương, epoch tốt nhất và epoch dừng. Bảng so sánh chính M3/M4 giữ kết quả theo protocol ban đầu cho đủ ba seed. Do chọn mẫu theo kết quả ban đầu, phụ lục chỉ trả lời liệu các trường hợp đó có thể học muộn. Launcher: `bash experiments/pvsg_trd_v1/min50_followup/launch.sh train`. Bảng riêng được ghi vào `reports/pvsg-trd-min50-followup-v1/validation_followup.json`.

### Bàn giao M5 · Đánh giá held-out

Sau khi 7 lượt kiểm tra hội tụ muộn hoàn tất, chạy `bash experiments/pvsg_trd_v1/m5/launch.sh evaluate` từ gốc repo. Dùng `bash experiments/pvsg_trd_v1/m5/launch.sh check` để kiểm tra trên CPU trong lúc chờ. Launcher chấm tuần tự 24 checkpoint gốc và 7 checkpoint min50, mỗi lần một model, batch inference 1 cửa sổ, dùng best.pt theo internal validation và seed đã train. 62 video theo manifest M2, toàn bộ cặp đồng hiện, window 64/stride 32, decoder 0,5. Hash checkpoint, danh sách seed, 62 ID và code M5 được khóa trong frozen.json trước inference. Chạy lại cùng lệnh để tiếp tục từ checkpoint đã chấm xong.

Đầu ra reports/pvsg-trd-m5-v1/ gồm report.md, summary.json, metric chi tiết theo run/domain/predicate, và timeline của tối đa năm GT tuple đúng + năm bỏ sót (Static GAT seed 17, thứ tự video/key). Head/mid/tail mỗi nhóm 19 predicates, xếp theo số tuple của 287 video fit, hòa thì theo predicate ID. Bảng chính giữ nguyên tám mô hình × ba seed. Phụ lục trình bày bảy cặp gốc/min50 riêng; việc chọn các trường hợp này dựa trên validation sau thí nghiệm chính.
