# Image → Scene Graph v1: kiểm tra CPU ngày 2026-11-15

**Đã triển khai và kiểm pipeline; chưa chạy lượt inference chính trên đủ 30 ảnh,
chưa có kiểm duyệt tay. Không chạy GPU, không huấn luyện, không push.**

Hướng dẫn và lệnh bàn giao: [LEVEL3_IMAGE_GRAPH_V1.md](../../docs/LEVEL3_IMAGE_GRAPH_V1.md).

## Kiểm tra tự động

Lệnh cuối:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q \
  tests/level3 tests/editing/test_operations.py \
  tests/editing/test_emnlp2020_graph_modification_v1.py \
  tests/editing/test_emnlp2020_task_b_v2.py
```

**121 passed**, gồm 31 case mới cho Image Graph và kiểm hồi quy benchmark
Level 3, EditGraph và representation/codec Task B. Ruff, Black và `bash -n`
launcher pass. Test mock đủ 30 ảnh chỉ kiểm luồng điều khiển/file/schema, không
được tính là inference thật hay chất lượng model.

Preflight CPU xác minh **30 ảnh thật, 15 Sidewalk +15 Travel**, bytes/hộp kích
thước, ID/task mapping, upstream code pinned và hai checkpoint thị giác. Hash:

- Manifest: `25d2a89119c97c7c850a456c4b5702f7b45d9860c4f4de1d45120a55219595be`.
- Tasks: `8a4560961499080b0444bac1e8ce05a28e949edf946e9e72e2c2a68d0c234da8`.
- Task B GNN seed42: `7ed060052782898ca9ba058ea4639a2ed91d666839a83c15d98ae2205ff6a0f2`.

Hai hash cohort trên được pin trong config mới; không sửa manifest đã khóa.
Đối chiếu SHA256 trước/sau của **1.269 tệp có sẵn** (tracked files tại trạng thái
người dùng đang có và toàn bộ `pilot/locked/`): không tệp nào đổi. `AGENTS.md` và
notebook cá nhân giữ nguyên thay đổi có sẵn của người dùng, không vào commit này.

## Smoke thật, hoàn toàn CPU

Đã tải checkpoint VG qua link chính thức của tác giả, kiểm SHA256 và nạp
`strict=True` thành công: RelTR có **63.901.928 parameters**, toàn bộ frozen.
Smoke dùng cùng resize 800/ngưỡng/seed với lệnh inference chính, không dùng
trọng số random hoặc output giả. Có hai lần kiểm cùng ảnh đầu; lần mới nhất:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_image_graph_v1 smoke \
  --output runs/level3/image-graph-v1/smoke-cpu-v2
```

| Hạng mục | Quan sát |
|---|---|
| Candidate | `sidewalk-2a7fa2d343bdf7` |
| Device / batch / seed | CPU / 1 / 42 |
| Graph ứng viên | 24 node, 2 directed edge |
| Thuộc tính đề xuất | 7 giá trị trên 6 node, cần duyệt tay |
| Receipt | 10 artifact có hash; `pilot_inference_complete=false` |
| Human review | 0; tất cả pending |

`candidates.json` của smoke có SHA256
`382cca7c3baedaa1de4cc078420179634a885e2a674f57a2425cdf97d7f4fe08`.
Run lưu config và hash từng file code thực thi tại thời điểm smoke; các kiểm tra
contract/review bổ sung sau đó được kiểm lại bằng CPU tests và CLI. Không đưa
graph/ảnh/checkpoint vào Git.

Môi trường: torch `2.7.0+cu128`, torchvision distribution `0.22.0` (module
`0.22.0+cu126`), Pillow `11.2.1`, scipy `1.15.3`, clip `1.0`. Có CUDA trong tên
binary không đồng nghĩa chạy GPU: smoke bị cố định `device=cpu`, đặt
`CUDA_VISIBLE_DEVICES=''`.

Viewer được mở bằng Chromium headless với `--disable-gpu`: không lỗi JavaScript;
hiển thị 24 hộp, chọn riêng node còn 1 hộp, hiển thị 2 cạnh có hướng, tải JSON và
nạp lại phiếu thành công. CLI nhập phiếu pending thật của smoke xuất **0 accepted
/1 pending**, không tự phê duyệt. Unit tests kiểm sửa node/attribute, lưu diff,
confidence mới `null`, từ chối phiếu stale/dangling/bbox lỗi/thiếu người duyệt.

## Kiểm Task B trên checkpoint thật

Đọc CPU weights/config/vocabulary, không forward model Task B. Embedding identity
khớp config **7 node /42 cạnh**; decoder có **4.758 lexical tokens** cộng bốn
token đặc biệt. Graph smoke:

- Pass schema và roundtrip codec, nhưng **24 >7 node**.
- Cả hai cạnh thị giác có hướng một chiều, khác reciprocal arcs của nguồn train.
- Sáu node có inline attributes; policy A nguồn train giữ attribute node + `att`.
- Ba payload OOV của decoder: `["pink"]`, `["plastic"]`, `["plastic","yellow"]`.

Kết luận `semantic_compatibility=not_established`, `direct_model_use_approved=false`.
Không flatten attribute, thêm cạnh ngược, truncate node hay chạy Task B để che
khác biệt. OOV nêu trên không phải OOV CLIP input và chưa cho biết target edits.

Số lượng graph/thuộc tính trên chỉ chứng minh pipeline thực thi, không phải
precision/recall hay kết quả benchmark. Thuộc tính CLIP có dự đoán sai rõ như
`plastic` trên `street`; người duyệt cần sửa/loại, và báo cáo độ tin cậy sau khi
duyệt pilot. Chưa đo tốc độ/VRAM GPU; chưa xử lý correspondence trước/sau chỉnh
sửa, image editing hay consistency scoring của toàn nhiệm vụ 3.
