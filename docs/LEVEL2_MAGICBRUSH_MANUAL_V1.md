# MagicBrush: giao diện gán graph bằng tay v1

Giao diện chạy cục bộ, dùng ảnh thật và manifest đã có. Không gọi model,
không GPU, không API trả phí. Mã riêng tại `tools/magicbrush_manual_v1/`,
không sửa `src/`, manifest/split cũ, nhãn tự động hoặc notebook cá nhân.

## Mở giao diện

Từ repo root:

```bash
myenv/bin/python -m tools.magicbrush_manual_v1 --stage plan
myenv/bin/python -m tools.magicbrush_manual_v1 --port 8776
```

Mở `http://127.0.0.1:8776/`. Server chỉ bind loopback; nếu chạy trên máy chủ
khác, dùng port forwarding của môi trường phát triển để truy cập cổng 8776.
Ctrl+C dừng server; dữ liệu đã lưu vẫn còn trên ổ đĩa.

Mặc định mở đúng 20 ID train đã chọn trước ở manifest MagicBrush, không lựa
lại theo chất lượng. Có thể chuyển sang toàn bộ **415 train +97 validation**.
90 mẫu test nội bộ vẫn được giữ trong manifest gốc và không xuất hiện trong
API/giao diện. Không đổi split hoặc gộp nhóm ảnh; giữ source group/session và
raw/RGB SHA như bộ đã khóa. Chưa tự gán hoặc đánh dấu hoàn thành mẫu thật nào.

## Cách điền một cặp

1. Nhập tên người gán. Ở **Ảnh nguồn**, bấm lên ảnh hoặc “Thêm đối tượng”;
   mỗi vật thể phân biệt là một dòng. Ví dụ `person`, `cup`, thuộc tính của
   cup là `red, ceramic`. Có thể đánh dấu điểm trên ảnh để nhớ vật nào;
   không bắt buộc vẽ bounding box.
2. Điền **quan hệ có hướng** bằng hai danh sách chọn và ô quan hệ:
   `person → holding → cup`. Mỗi relation là một dòng; cho phép gõ nhãn mới
   ngoài các gợi ý. Gợi ý là từ điển tĩnh, không học từ val/test.
3. Nhấn **Xác nhận nguồn**. Nguồn phải có tên đối tượng, quan hệ và endpoint
   hợp lệ, trong budget. Server ghi riêng bằng chứng source trước khi cho
   xem ảnh đích/câu yêu cầu; chuẩn hóa source ID chỉ từ graph nguồn. Graph
   nguồn được sao chép sang đích, giữ ID của đối tượng và cạnh.
4. Ở **Ảnh đích**, sửa graph sao cho khớp ảnh. Đổi màu/thuộc tính hoặc thay
   loại vật thể tương ứng: sửa dòng cũ. Thêm vật thể độc lập: thêm dòng mới.
   Vật thể mất đi: xóa dòng; giao diện báo số quan hệ liên quan sẽ bị xóa và
   cho hoàn tác. Sửa predicate trên dòng quan hệ cũ biểu thị thay quan hệ;
   xóa dòng rồi thêm dòng mới biểu thị delete+add. Không tự ghép hai vật thể
   chỉ vì cùng tên; điểm đánh dấu giúp phân biệt từng instance.
5. Nhấn **Xem thay đổi & kiểm tra**. Server dùng nguyên
   `diff_aligned_graphs`, `apply_operations`, canonical ID policy và action
   codec hiện có. Xem thao tác được diễn giải bằng tiếng Việt, đối chiếu hai
   ảnh và xác nhận hai mục về câu yêu cầu/bảo toàn phần không đổi.
6. **Hoàn thành & cặp tiếp** lưu nhãn candidate và mở cặp kế tiếp. Nhãn vẫn
   `human_review_status=pending`; hoàn thành gán không phải review độc lập.

Quy tắc mô tả: dùng nhãn tiếng Anh nhất quán; một vật thể mỗi dòng dù cùng
tên; thuộc tính cách bằng dấu phẩy; giữ các vật thể nổi bật không bị chỉnh.
Không cần mô tả mọi pixel hoặc ép chi tiết khó thấy vào graph. Chốt cùng mức
chi tiết cho cả nguồn và đích. Nếu ảnh đích không khớp câu lệnh, sửa toàn cảnh
ngoài schema hoặc identity không rõ, chọn **Tạm bỏ qua** và ghi lý do; không
gán thay đổi theo câu lệnh khi ảnh không thể hiện nó.

Ảnh và câu yêu cầu bị ẩn trước xác nhận nguồn cả ở API, không chỉ bằng CSS.
Điều này giữ quy trình source-first trong công cụ, không chứng minh người
gán chưa từng thấy ảnh đích ở nơi khác. “Sửa lại nguồn” mở lượt mới và bỏ graph
đích của lượt đang thao tác; mọi event cũ vẫn được giữ để đối chiếu.

## Tiện ích và lưu dữ liệu

- Source: ảnh giữ bên cạnh bảng điền sau khi thêm vật đầu tiên. Target:
  hai ảnh và hai graph cạnh nhau trên desktop; màn hình nhỏ xếp dọc.
- Gợi ý tên vật, thuộc tính, quan hệ; gõ giá trị mới vẫn được.
- Tìm mã cặp, lọc train/validation, chưa xong/đã xong/tạm bỏ qua, thanh tiến độ.
- Phóng to ảnh, điểm đánh dấu theo từng vật thể, undo/redo graph và xóa cạnh
  liên quan có xác nhận. ID nội bộ chỉ dùng liên kết, không phải nhãn ngữ nghĩa.
- `N`: thêm đối tượng; `R`: thêm quan hệ; `Ctrl/Cmd+S`: lưu;
  `Ctrl/Cmd+Z`: hoàn tác graph khi không nhập văn bản; `←/→`: đổi cặp khi
  không nhập; `?`: hướng dẫn. Trong ô văn bản, undo của trình duyệt giữ nguyên.
- Tự lưu sau 700 ms ngừng nhập; bản nháp chưa đủ nhãn vẫn được lưu. Browser
  giữ thêm bản nháp phục hồi khi gián đoạn. Phải thấy “Đã lưu vào máy” trước
  khi coi dữ liệu đã ghi server. Mất kết nối vẫn có thể tải bản nháp JSON.
- **Tải bản nháp / Khôi phục** hỗ trợ JSON. Khôi phục chỉ nhận đúng workspace,
  ID và revision hiện tại; bản nháp cũ không tự ghi đè bản mới. Xung đột giữa
  hai tab giữ nội dung đang nhập, yêu cầu đối chiếu; không last-write-wins.
- Source đã khóa không được autosave sửa ngầm. Sửa một mẫu đã hoàn thành sẽ
  chuyển về nháp đích và phải xác nhận hoàn thành lại.

Output riêng: `data/processed/level2/magicbrush-manual-v1/`:

```text
workspace.json                 # provenance manifest gốc và protocol
events/<hash-ID>/000001-*.json   # lịch sử append-only, parent hash, timestamp
exports/<snapshot-hash>/        # manifest.json, edits.json, report.json
```

Mỗi event ghi annotator, source lock, graph/điểm/ghi chú, trạng thái, revision,
hash parent/workspace và hash code giao diện/backend. Atomic writes không
ghi đè event; file lock và kiểm revision ngăn hai server cùng ghi mất nhãn.
Không cần lưu ảnh mới; chỉ phục vụ file nằm trong manifest, kiểm raw SHA và
không cho truy cập đường dẫn tùy ý. Dữ liệu annotation không commit vào Git.

## Xuất nhãn và giới hạn

Nút **Xuất nhãn đã xong** tải JSON bundle và lưu snapshot riêng. CLI tương
đương:

```bash
myenv/bin/python -m tools.magicbrush_manual_v1 --stage compile
```

Chỉ mẫu `completed` hợp lệ vào `edits.json`; nháp/defer vẫn ở lịch sử và báo
cáo số lượng. Snapshot giữ cùng ID/split/provenance ảnh, trả nhãn theo schema
loader Task B hiện có (`load_examples`), với `label_origin=human_image_annotation`.
Budget: mỗi graph 1–20 node, ≤60 edge, 1–12 operations, ≤2048 action tokens.
No-op hoặc giao dịch/codec sai không được hoàn thành; không hạ budget để tăng
coverage. Điểm đánh dấu phục vụ annotation, không trở thành feature model.

Export sớm dưới 300 mẫu phục vụ backup/kiểm nhãn; luôn ghi
`ready_for_training=false`, `independent_review_completed=false`. Cần đủ
≥300 cặp train+val dùng được, ≥20 kiểm độc lập và báo độ tin cậy, rồi khóa
protocol/config huấn luyện riêng. Gán tay không thay thế kiểm chéo. Hạ tầng
packet review cũ đọc được schema xuất, nhưng mô tả nguồn nhãn trong viewer /
launcher cũ còn ghi VLM/pseudo-label; cần phiên bản review/launcher cho nguồn
manual trước khi dùng huấn luyện chính, không sửa các experiment đã khóa.

## Kiểm chứng

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q tests/editing/test_magicbrush_manual_v1.py
```

Tests chỉ gán nhãn trên fixture tổng hợp trong thư mục tạm; không tạo người
kiểm hoặc nhãn giả cho ảnh thật. Browser kiểm source-first, điểm trên ảnh,
nhập tiếp sau autosave, correspondence, diff, hoàn thành/export, mobile,
undo cascade, khôi phục và xung đột. Kiểm actual manifest/ảnh chỉ đọc; không
coi smoke test giao diện là manual annotation hay manual review.
