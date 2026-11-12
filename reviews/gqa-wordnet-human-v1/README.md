# GQA WordNet — giao diện human review

Giao diện cho đúng **30 nhãn** trong
[`review_sample.csv`](../../reports/level2/gqa-wordnet-v1/review_sample.csv).
Đã chuẩn bị **150 ví dụ, 5 ảnh train khác nhau cho mỗi nhãn**. Mỗi ví dụ có
ảnh toàn cảnh với bounding box đỏ và crop của đúng instance. Chưa có quyết
định review nào được tạo sẵn; chọn synset cũng không tự chuyển thành accept.

## Mở và sử dụng

Từ thư mục gốc repository:

```bash
myenv/bin/python -m tools.prepare_gqa_wordnet_review serve
```

Mở **http://127.0.0.1:8765/** trên máy chạy repo. Nếu chạy repo qua SSH, có thể
forward cổng 8765 bằng `ssh -L 8765:127.0.0.1:8765 <máy-chạy-repo>`.
Server chỉ nghe loopback và chỉ phục vụ viewer/các asset được tạo, không phục
vụ checkpoint hoặc các thư mục khác của repo. Dùng `--port` nếu cần cổng khác.
`Ctrl+C` dừng server.

Cũng có thể mở trực tiếp [`index.html`](index.html) bằng trình duyệt. Chế độ
file hoạt động không cần server; nút **Lưu CSV vào reviews/** sẽ tắt, còn
**Xuất CSV** vẫn tải được file xuống máy đang mở trình duyệt.

1. Chọn nhãn ở thanh bên. Đối chiếu nhãn, trạng thái mapping, synset hiện tại,
   tất cả ứng viên và định nghĩa WordNet với **cả 5 ảnh**. Bấm ảnh để xem lớn.
2. Chọn synset phù hợp và **Accept**, hoặc **Defer** nếu thiếu căn cứ, không
   có ứng viên phù hợp hay các ảnh thể hiện những nghĩa khác nhau. Ghi rõ vấn
   đề trong ghi chú. Nhãn/box là annotation gốc, có thể có lỗi cần ghi nhận.
3. Điền tên người thực sự duyệt. Ô tên mặc định giúp điền nhanh cho nhãn tiếp
   theo; tên người duyệt của từng nhãn vẫn được lưu riêng. Không tự nhận mọi
   mapping kế thừa VG150 là mapping đã được kiểm tra cho GQA.
4. Dùng **Xuất CSV** để tải file, hoặc **Lưu CSV vào reviews/** khi chạy server.
   Server tạo file mới `submissions/review-<hash-nội-dung>.csv` trong thư mục
   này, không ghi đè bản khác. Lưu cùng nội dung sẽ xác minh và dùng lại file.

Bản nháp được lưu trong localStorage theo fingerprint của packet. **Nhập CSV**
khôi phục kết quả trên trình duyệt khác; chỉ nhận CSV xuất từ cùng packet,
đúng ID/nhãn/metadata/bằng chứng. Import lỗi giữ nguyên bản nháp đang có.
Nên xuất hoặc lưu CSV trước khi đóng trang, nhất là khi trình duyệt chặn
localStorage cho file cục bộ.

CSV giữ nguyên thứ tự và giá trị `class_id`, `label`, trạng thái/synset hiện
tại, định nghĩa và ứng viên từ CSV nguồn. Các cột quyết định là `decision`
(`accept`, `defer` hoặc để trống), `selected_synset`, `reviewer`, `notes`.
Thêm `train_image_ids`, `evidence_status`, `review_packet_sha256` để đối chiếu
provenance. Nhãn chưa review vẫn để trống; accept cần chọn ứng viên và có tên
người duyệt. Nhãn không có ứng viên không thể accept trong viewer này; dùng
defer và đề xuất nghĩa bổ sung trong ghi chú.

Viewer chỉ thu thập review. CSV xuất ra **chưa được tự áp dụng** vào mapping,
KG hoặc embedding; chưa có bước compile/approval cho thực nghiệm chính.

## Các trường hợp cần chú ý

**8 nhãn đa nghĩa:** bacon (56), burner (217), caramel (267), cream (421),
display (472), dumplings (516), sea (1269), terminal (1502).

**7 nhãn chưa có ứng viên WordNet trong mapping hiện tại:** food truck (604),
hair clip (682), mexican food (899), mouse pad (932), picnic table (1091),
pizza slice (1125), touchpad (1542). Đây là thiếu hỗ trợ về mapping; cả bảy
vẫn có đủ 5 ảnh thật. Không tự thay chúng bằng khái niệm gần nghĩa.

15 nhãn còn lại gồm 8 auto-link và 7 lựa chọn kế thừa VG150. Toàn bộ 30 nhãn
vẫn cần quyết định mới của người duyệt trong ngữ cảnh GQA.

## Tái tạo viewer

```bash
myenv/bin/python -m tools.prepare_gqa_wordnet_review build
```

Mặc định đọc sample CSV, mapping và manifest GQA đã khóa, cache
`data/cache/gqa-full-clip-v1/`, cùng ZIP ảnh
`data/raw/gqa/archives/images.zip`. Cần NLTK WordNet 3.0 có sẵn; không tải thêm
corpus, không chạy encoder hoặc huấn luyện.

Chỉ mở **274 train HDF5 shard**, kiểm hash từng shard và quét đủ **69.942 ảnh
train**. Không mở annotation JSON hoặc shard validation/test. Đọc danh sách
ID split từ manifest để kiểm tra không trùng; các quyết định ngữ nghĩa chỉ
được hỗ trợ bằng record object/box của train. Predicate, thuộc tính và visual
feature không tham gia lấy mẫu hay hiển thị.

Mỗi nhãn: lấy 5 image ID có SHA256 nhỏ nhất của
`gqa-wordnet-human-v1:2026:class_id:image_id`. Trong một ảnh chọn matching
instance có diện tích lớn nhất; nếu bằng nhau chọn object ID nhỏ nhất.
Tọa độ annotation được scale theo kích thước JPEG thật, rồi vẽ box và crop
bằng floor/ceil như workflow Level 1. Không sửa pixel ngữ nghĩa hoặc tạo ảnh
giả. Đọc member từ ZIP kiểm CRC và lưu SHA256 nguồn/asset; không tính lại
hash toàn ZIP 22 GB mỗi lần build. Hash archive khai báo trong cache được
giữ riêng trong provenance.

Asset ở `data/knowledge/gqa-wordnet-human-v1/images/` (300 JPEG, khoảng 13,6 MB),
được Git ignore; cần giữ thư mục này cùng repo để mở viewer. Manifest và
viewer ở đây được theo dõi trong Git; không commit ảnh gốc hoặc checkpoint.
[`review-data.json`](review-data.json) chứa ID ảnh/object, box annotation/JPEG,
kích thước, ZIP member, checksum, ứng viên/định nghĩa, các lỗi và nguồn đã đọc.

Build lại cùng môi trường/nội dung là idempotent và không chạm `submissions/`.
Nếu output khác nội dung, CLI từ chối ghi đè. Khi đổi seed/số ảnh/nguồn, dùng
thư mục phiên bản mới, ví dụ:

```bash
myenv/bin/python -m tools.prepare_gqa_wordnet_review build \
  --seed 2027 --per-class 3 \
  --output reviews/gqa-wordnet-human-v2 \
  --assets data/knowledge/gqa-wordnet-human-v2/images

myenv/bin/python -m tools.prepare_gqa_wordnet_review serve \
  --output reviews/gqa-wordnet-human-v2 \
  --assets data/knowledge/gqa-wordnet-human-v2/images
```

## Kiểm chứng

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q \
  tests/knowledge/test_gqa_human_review.py \
  tests/knowledge/test_gqa_human_review_browser.py \
  tests/knowledge/test_prepare_wordnet_review_examples.py \
  tests/knowledge/test_prepare_level1_review.py \
  tests/knowledge/test_entity_linking.py tests/knowledge/test_sense_review.py
```

Browser tests dùng fixture giả trong thư mục tạm, không ghi quyết định vào
bộ GQA thật. Chúng kiểm export/import CSV có Unicode/quote/newline, kiểm ID,
defer không có ứng viên, tên người duyệt, bản nháp, lưu file và mobile/file mode.
Playwright là phụ thuộc tùy chọn cho QA, không cần để tạo/mở viewer. Mốc này
đã chạy Chromium qua Playwright 1.65.1; nếu chưa cài môi trường browser, hai
test browser sẽ skip. Có thể cài bằng `myenv/bin/python -m pip install
playwright==1.65.1` rồi `myenv/bin/python -m playwright install --with-deps chromium`.

Đã kiểm trực tiếp cả 30 nhãn/300 asset trên Chromium, đủ toàn bộ radio ứng
viên, không có lỗi JavaScript và không tràn ngang ở viewport 390 px. Không
tạo review thật từ quá trình QA. [Biên bản hash và dữ liệu](verification.json)
xác nhận các artifact thí nghiệm trước vẫn giữ nguyên. Chưa chạy training mới.
