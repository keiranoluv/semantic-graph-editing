# Level 3 — Image → Scene Graph cho pilot Việt Nam v2

Triển khai riêng tại `tools/level3_image_graph_v1/`, không đổi `src/`, Level 1/2,
manifest/config cũ hoặc 30 ảnh đã khóa. Đây là tầng sinh **graph ứng viên** và
kiểm duyệt; chưa phải nhãn chuẩn, chưa sinh ảnh chỉnh sửa hay chấm consistency.

## Mô hình và phạm vi

- **RelTR VG / ResNet-50**: checkpoint công khai của tác giả, 150 nhãn đối tượng
  và 50 predicate; mã upstream có thêm lớp background và head có thêm no-object.
  Chạy `eval()`, `requires_grad_(False)`, `torch.inference_mode()`, không optimizer.
- **OpenAI CLIP ViT-B/32** đóng băng: đề xuất thuộc tính trên crop từ hộp RelTR.
  RelTR không có attribute head. Thuộc tính trong phiên bản này chỉ gồm màu và
  chất liệu trong vocabulary đã khóa; không tuyên bố phát hiện đủ mọi thuộc tính.
- Chạy tuần tự **RelTR trên 30 ảnh → giải phóng model → CLIP trên các crop**.
  Batch ảnh/crop **1**, seed **42**, resize RelTR cạnh ngắn **800**, chuẩn hóa
  ImageNet theo demo chính thức. Dùng pixel lưu trong ảnh, không xoay EXIF;
  viewer dùng cùng hệ tọa độ. Không train, không có hàng đợi thí nghiệm tiếp theo.
- Không dùng PredCls Level 1/PE-Net thay cho bộ đọc ảnh: các pipeline đó cần
  box/nhãn cho sẵn. Không thay checkpoint hay cache Level 1/2; chỉ đọc CLIP cũ.

Nguồn: [RelTR chính thức](https://github.com/yrcong/RelTR),
[inference tại commit đã khóa](https://github.com/yrcong/RelTR/blob/fca7397e9aaeccd95541e83afa4b971f3fa89014/inference.py),
[checkpoint VG chính thức](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view),
[CLIP chính thức](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6).
Code gọi các module inference upstream, tự dựng ResNet với `weights=None` rồi
nạp đầy đủ checkpoint bằng `strict=True`, tránh tải thừa ImageNet và dựng loss.
Không sửa source upstream, không cần biên dịch extension đánh giá/training.

| Asset | Danh tính khóa |
|---|---|
| RelTR code | `fca7397e9aaeccd95541e83afa4b971f3fa89014` |
| RelTR checkpoint SHA256 | `2b3601e5cc5d835d9538d78b698dd7b3f4f1646bdb0e5c459d2cbe959d5906b1` |
| CLIP checkpoint SHA256 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

SHA checkpoint RelTR được tính từ tệp tải qua link của tác giả trong phiên này,
không phải hash do tác giả công bố. `prepare` kiểm hash đó; không tự chấp nhận
checkpoint khác. Kiểm code upstream bằng byte từng file so với Git tree pinned,
không chỉ kiểm tên revision. Không tải/upload ảnh lên API bên ngoài.

## Chạy inference trên 30 ảnh thật

Trong workspace hiện tại, asset RelTR đã chuẩn bị; CLIP và Task B checkpoint đã
có. Các lệnh sau chạy từ root repository:

```bash
# Chỉ cần khi dựng môi trường: dùng môi trường phù hợp với torch/torchvision hiện có.
myenv/bin/python -m pip install -r requirements-level3-image-graph-v1.txt

# CPU/network: tải code/checkpoint công khai nếu thiếu, không inference.
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_image_graph_v1 prepare

# CPU: xác minh đủ 30 ảnh, 15/15, tasks, code, checkpoint/vocab, cấu hình Task B.
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_image_graph_v1 preflight

# GPU: NGƯỜI DÙNG tự chạy sau commit code và CPU checks.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_level3_image_graph_v1.sh
```

Output mặc định: **`runs/level3/image-graph-v1/pilot30/`**. Launcher chỉ chạy một
job: RelTR rồi CLIP như trên, không tự chạy Task B, editor, benchmark hoặc lượt
khác. `flock` chặn hai launcher này chạy đồng thời. Không có huấn luyện nên không
mở W&B training run. Chưa đo VRAM/tốc độ GPU; không có cam kết từ smoke CPU.

Output đã tồn tại sẽ bị từ chối, kể cả run bị ngắt. Để chạy lại, giữ nguyên run cũ
và chọn directory mới; không có resume hoặc tự bỏ qua ảnh lỗi:

```bash
CUDA_VISIBLE_DEVICES=0 OUTPUT=runs/level3/image-graph-v1/pilot30-r2 \
  bash scripts/run_level3_image_graph_v1.sh
```

CLI tương đương, cũng hỗ trợ CPU nếu cần:

```bash
myenv/bin/python -m tools.level3_image_graph_v1 infer \
  --device cuda --output runs/level3/image-graph-v1/pilot30-r3
```

`infer` luôn kiểm đủ cohort rồi chạy đúng 30 ảnh, không có `--limit` để vô tình
coi subset là pilot hoàn tất. `smoke` là lệnh riêng chỉ chạy ảnh đầu bằng CPU:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_image_graph_v1 smoke \
  --output runs/level3/image-graph-v1/smoke-cpu-new
```

Tất cả subcommand có `--help`. Các đường dẫn asset có thể đổi bằng
`--reltr-repo`, `--checkpoint`, `--clip-weights`, `--task-b-checkpoint`, nhưng
weights thị giác vẫn phải khớp hash. Task B mặc định dùng
`runs/level2/emnlp2020-task-b-wandb-v4/gnn-seed42/best.pt`, chỉ đọc trên CPU để
kiểm vocabulary/cấu hình/kích thước weights; không forward model chỉnh graph.

## Schema, lọc và provenance

`source_graph` có đúng hai key `nodes`, `edges`, được roundtrip qua
`src.editing.operations.EditGraph`:

```json
{
  "nodes": [{"id": "n001", "label": "shirt", "attributes": ["red"]}],
  "edges": []
}
```

Ví dụ schema trên chỉ để minh họa, không phải kết quả inference. Metadata nằm
ngoài graph trong `grounding.nodes[id]` / `grounding.edges[id]`:

- Node: `bbox_xyxy` theo pixel ảnh gốc, confidence softmax RelTR, raw query IDs,
  thuộc tính đã chọn và bằng chứng CLIP. Confidence không được coi là calibrated.
- Edge: ID, `subject → object` trong graph, confidence predicate và tích
  `P(subject) × P(predicate) × P(object)` trong `triplet_score`.
- Thuộc tính: prompt, family, cosine gốc, `confidence=(cosine+1)/2`,
  `confidence_kind=rescaled_cosine_not_probability`. Đây là similarity, không
  phải xác suất đúng hay điểm attribute head của RelTR.
- Record: candidate ID = `image_id` từ manifest khóa, SHA256 bytes ảnh, toàn bộ
  provenance nguồn ảnh, hash raw prediction và hash run. ID node chỉ có ý nghĩa
  **trong ảnh này**; không dùng ID trùng tên để ghép trước/sau chỉnh sửa.

Giữ argmax trên toàn head, bỏ background/no-object; không cưỡng ép chọn lớp
foreground. Ngưỡng object/predicate 0,3. Gộp box cùng nhãn khi IoU ≥0,7 theo thứ
tự confidence giảm, tie-break query ID; giữ bbox/confidence đại diện cao nhất
và mọi query ID đã gộp. Không gộp chỉ vì cùng nhãn. Gộp triple trùng, giữ hướng;
bỏ self-edge sinh ra do gộp endpoint. Lưu raw logits/boxes và các hypothesis bị
bỏ để audit. Có thể còn node trùng khác nhãn hoặc box lệch; cần duyệt tay.

CLIP chọn tối đa một giá trị mỗi family nếu cosine ≥0,2 và margin với giá trị
thứ hai ≥0,02, nếu không thì abstain. Các ngưỡng/vocab này là cấu hình khởi đầu,
không phải kết quả hiệu chỉnh trên pilot. `attributes=[]` nghĩa là không có
thuộc tính được chọn, không chứng minh đối tượng không có thuộc tính. CLIP có
thể gán chất liệu/màu sai do crop chứa nền, vật nhỏ hoặc vocabulary không có đáp
án; ví dụ smoke có đề xuất `plastic` cho `street`, cần loại trong review.

Không cắt bớt node/cạnh để vừa Task B. Graph rỗng có trạng thái
`empty_prediction_needs_review`, không được tự coi là quan sát hợp lệ và không
được human-accept khi vẫn rỗng.

## Artifact và kiểm duyệt

| Artifact | Nội dung |
|---|---|
| `run-start.json`, `vocabulary.json` | config, seed, model/code/weight hashes, package versions, commit, cohort hashes |
| `raw/<candidate>.json` | logits/boxes nguyên đầu ra và hypothesis RelTR đã decode |
| `attributes/<candidate>.json` | mọi nhãn/prompt/score CLIP, kể cả không được chọn |
| `candidates.json` | graph ứng viên + grounding + provenance |
| `source_graph_tasks.jsonl` | 30 source tasks mới, nối về manifest cũ bằng ID/hash |
| `task-b-compatibility.json` | kiểm schema, giới hạn thật, OOV, directedness, attribute policy |
| `review.html`, `<candidate>.svg` | viewer offline: ảnh/hộp, mũi tên quan hệ, JSON có thể sửa |
| `review-template.json` | phiếu pending, ràng buộc hash với từng graph gốc |
| `receipt.json` | chỉ xuất cuối run; hash các artifact, số ảnh và cờ pilot hoàn tất |

Mở `review.html` bằng trình duyệt cục bộ. Chọn từng đối tượng để nhìn riêng hộp;
xem graph có hướng và confidence/provenance. Sửa `source_graph` và `boxes` trong
JSON. Giữ ID đối tượng không đổi; khi thêm dùng ID mới, khi xóa sửa cả cạnh liên
quan. Điền người duyệt và ghi chú; kiểm cả bốn mục đối tượng, identity/box,
attribute, quan hệ có hướng. Chọn `accept`, `reject` hoặc giữ `pending`. Tải
phiếu JSON; có thể nạp lại phiếu khi tiếp tục duyệt. Chỉ thao tác trên trang
không tự ghi xuống đĩa. SVG graph ban đầu vẫn là bằng chứng ứng viên gốc; hộp
preview cập nhật theo JSON sửa.

Nhập phiếu, ví dụ lưu tệp tải về trong `runs/level3/image-graph-v1/`:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_image_graph_v1 review \
  --bundle runs/level3/image-graph-v1/pilot30 \
  --submission runs/level3/image-graph-v1/level3-image-graph-review.json \
  --output runs/level3/image-graph-v1/reviewed-v1
```

CLI kiểm receipt, hash từng record, đủ ID không trùng, graph không dangling,
bbox hợp lệ, reviewer/checks; chỉ xuất record `accept` vào `reviewed.json` và
`source_graph_tasks.jsonl` mới. Lưu quyết định của cả cohort, diff thao tác và
hash phiếu. Thành phần do người sửa có `confidence=null`, không mượn score của
model cho nội dung mới. `pending/reject` không xuất như graph đã chấp nhận.
Graph ứng viên và locked dataset luôn nguyên vẹn. Phiếu giả trong unit tests
không phải human review thật. Không tự đánh dấu đủ 30 ảnh đã duyệt.

Viewer nhúng thumbnail ảnh nguồn để chạy offline; giữ artifact cục bộ theo
provenance-only policy hiện tại. Chưa có giấy phép/privacy clearance từng ảnh;
việc sinh graph không thay đổi trạng thái đó.

## Task B: kết luận tương thích

Đọc trực tiếp `alignment.py`, `compiler.py`, `ActionVocabulary`, `feature_texts`,
encoder/collator và checkpoint EMNLP2020 v4 hiện có:

| Khía cạnh | Graph thị giác mới | Task B đã train | Kết luận |
|---|---|---|---|
| Container | `EditGraph` | `EditGraph` | hợp lệ cú pháp/codec |
| Thuộc tính | tuple trên node, có owner | node riêng + liên kết `att`; policy A không flatten | chưa tương thích ngữ nghĩa |
| Quan hệ | subject → object có hướng | dữ liệu vô hướng được giữ bằng hai arc đối ứng | không tự thêm cạnh ngược |
| Kích thước | số đối tượng thật sau lọc | checkpoint v4: 7 node / 42 cạnh | kiểm từng ảnh, không truncate |
| Decoder | nhãn/attribute arrays có thể mới | vocabulary lexical train-only | kiểm OOV từ checkpoint |
| Đầu vào CLIP text | nhãn/câu mở | cùng cơ chế encode nhưng domain khác | OOV decoder không phải OOV encoder |
| Grounding | bbox/confidence/provenance | encoder hiện tại không dùng bbox/confidence | không coi hỗ trợ định vị thị giác đã được học |

Canonical ID mapping chỉ để audit, không sửa graph. Vocab OOV hiện kiểm **payload
của graph nguồn nếu cần phát sinh lại**, chưa biết target edits nên không phải
trần exact-match của pilot. Cùng schema, không OOV, hay vừa ngân sách vẫn không
chứng minh chuyển domain. Mọi report giữ `direct_model_use_approved=false`,
`semantic_compatibility=not_established`.

Có thể chạy audit riêng mà không chạy model thị giác:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_image_graph_v1 compatibility \
  --bundle runs/level3/image-graph-v1/pilot30 \
  --output runs/level3/image-graph-v1/task-b-audit-r2.json
```

Cần một adapter representation có version, chính sách owner/direction/instance
identity minh bạch và cặp edit có kiểm tay để đánh giá transfer trước khi dùng
Task B làm tầng dự đoán edit. Không suy owner từ liên kết `att` vô hướng. Công
việc này không tự train model chỉnh graph hoặc thị giác mới.

## Kiểm tra và giới hạn theo đề

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/level3 tests/editing/test_operations.py
myenv/bin/python -m ruff check tools/level3_image_graph_v1 tests/level3/test_image_graph_v1.py
bash -n scripts/run_level3_image_graph_v1.sh
```

Xem [bằng chứng CPU/smoke](../reports/level3/image-graph-v1-validation.md). Unit
tests kiểm decode shape/background/direction, box, gộp instance, attribute
abstention, dữ liệu khóa, review/hash/tamper, compatibility và mock đủ 30 ảnh.
Mock không đo chất lượng thị giác. Smoke thật chỉ kiểm chạy end-to-end trên một
ảnh; cần review cả pilot trước khi báo độ tin cậy.

Đề bài trang 2–4 cho phép SGG tiền huấn luyện và yêu cầu kiểm tay độ tin cậy.
Pilot 30 ảnh vẫn chưa đạt **100 ảnh +150–200 yêu cầu**, chưa có graph đích, điểm
correctness/preservation, nhãn đạt/không đạt, Spearman, CLIPScore/LPIPS hay 10 ca
bất đồng. Dataset EMNLP Task B thay thế không tự đáp ứng yêu cầu gốc xây ≥300
cặp MagicBrush/InstructPix2Pix và kiểm tay ≥20 mẫu; cần giữ phần khó khăn,
đổi nguồn và giới hạn representation trong báo cáo, theo thỏa thuận của dự án.
