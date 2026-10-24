# Kiểm tra mapping WordNet cho mức 1

Mapping là bảng nối **class ID/nhãn đối tượng VG150** với một **nghĩa từ WordNet**.
Ví dụ class `airplane` có gợi ý `airplane.n.01`, nghĩa là máy bay cánh cố định.
Một từ có nhiều nghĩa cần chọn nghĩa phù hợp với đối tượng của dataset.
GAT+KG sẽ dùng embedding của nghĩa này cho đối tượng mang nhãn tương ứng.

Đủ 150 nhãn có gợi ý tự động, nhưng chỉ 124 gợi ý được đưa vào mapping draft
dùng cho KG/TransE; 26 gợi ý còn lại có trạng thái `deferred_unmapped` vì nghĩa
không rõ từ tên nhãn. Đây là coverage 124/150, không khẳng định 124 lựa chọn
đều đúng. Smoke đã xác minh code chạy được; chưa xác minh các lựa chọn nghĩa.
Nếu đổi mapping, cần tái tạo
KG/TransE/export và đánh giá lại, nên kiểm tra trước các lượt train dài sẽ tiết
kiệm thời gian hơn.

## Tạo bảng đọc bằng Excel/LibreOffice

```bash
python -m scripts.prepare_level1_review
```

Đã chạy trên máy project. Bảng nằm ở
`data/knowledge/level1-human-review-v2/review.csv`, UTF-8 BOM để Excel đọc được.
File `v1` được giữ nguyên để bảo toàn bản worksheet trước đây; `v2` hiển thị
thêm gợi ý/lý do của cả 26 dòng đang defer.
Không chạy lại vào cùng folder sau khi đã sửa CSV; dùng `--output-dir` mới
nếu cần tạo template khác. Script từ chối ghi đè nội dung đã thay đổi.

Các cột quan trọng:

| Cột | Ý nghĩa / cách điền |
|---|---|
| `class_id`, `label`, `status` | Nhận diện nhãn và trạng thái gốc; không sửa |
| `suggested_synset` | Nghĩa do bản draft AI gợi ý, chưa phải kết quả người kiểm tra |
| `suggested_definition` | Giải thích nghĩa đó bằng tiếng Anh từ WordNet |
| `all_candidates` | Các nghĩa hợp lệ trong mapping nguồn cùng định nghĩa |
| `draft_mapping_state` | `selected_provisional` cho 124 dòng đã dùng; `deferred_unmapped` cho 26 dòng chưa đưa vào KG |
| `deferred_reason` | Lý do AI không chọn nghĩa để train, dù có gợi ý tự động |
| `selected_synset` | Người kiểm tra điền nghĩa chọn sau khi đọc; có thể khác gợi ý nếu nằm trong candidates |
| `decision` | `accept` nếu chấp nhận nghĩa đã chọn, `defer` nếu chưa chắc; để trống nếu chưa xem |
| `reviewer` | Tên thành viên thực sự kiểm tra |
| `notes` | Lý do lựa chọn hoặc điều chưa chắc; nên ghi cho mọi nhãn |

## Xem ví dụ ảnh train cho nhãn mơ hồ

Codex đã chạy công cụ sau để lấy ví dụ cho **cả 150 nhãn**:

```bash
python -m scripts.prepare_wordnet_review_examples
```

Mỗi nhãn có 5 ảnh thuộc **train split**, xáo thứ tự bằng seed 2026 rồi lấy
ảnh đầu tiên tìm được cho nhãn đó. Danh sách nằm trong
`data/knowledge/level1-human-review-v2/train_examples_with_boxes.csv`.
Mỗi dòng ghi `class_id`, `label`, `image_id`, `object_index`, hộp bao pixel
`box_xyxy` và URL ảnh. Manifest cùng thư mục ghi hash split/HDF5/vocabulary.
Script chỉ đọc nhãn/hộp đối tượng, không đọc predicate hoặc quan hệ GT.

Nếu URL JPEG trong CSV không mở được từ máy nhóm, dùng archive ảnh ở
[Visual Genome](https://visualgenome.org/api/v0/api_home), tra JPEG theo
`image_id` và đối chiếu hộp. Trên máy Codex host ảnh trả lỗi kết nối, nên
Codex chưa xem trực tiếp các JPEG này.

Quy trình cho một dòng, ví dụ `bike` (class ID 11):

1. Mở `review.csv`, đọc `suggested_synset`, định nghĩa, các candidate khác và
   `deferred_reason`. Với `bike`, gợi ý là `bicycle.n.01` nhưng còn khả năng
   `motorcycle.n.01`.
2. Lọc `train_examples_with_boxes.csv` theo class ID 11, mở vài ảnh train và
   xem vật thể trong hộp bao. Chỉ dùng ảnh, nhãn và hộp; không xem quan hệ GT.
3. Nếu các ví dụ ủng hộ một nghĩa và reviewer tự tin, điền
   `selected_synset`, `decision=accept`, tên thật ở `reviewer`, và `notes`
   mô tả các image ID đã xem cùng lý do. Nếu ảnh dùng lẫn cả hai nghĩa hoặc
   chưa đủ bằng chứng, điền `decision=defer`, để trống `selected_synset` và
   ghi lý do ở `notes`.
4. Một người thứ hai nên xem độc lập những nhãn khó; lưu bất đồng và cách
   giải quyết. Không điền tên người nếu họ chưa thực sự xem.

Làm 26 dòng `deferred_unmapped` trước. Sau đó kiểm tra 101 dòng AI đã chọn và
23 dòng WordNet tự nối; “tự nối” vẫn chưa chứng minh đúng nghĩa của ảnh.
Không nhất thiết mở đủ 5 ảnh cho nhãn rõ ràng, nhưng cần ghi rõ đã kiểm tra
định nghĩa và bằng chứng nào. Ghi quyết định cá nhân cho đủ 150 dòng nếu muốn
đi qua quy trình reviewed; dòng nào chưa rõ vẫn giữ `defer`. Gate huấn luyện
reviewed hiện tại còn yêu cầu KG sẵn sàng, nên có dòng `defer` thì chưa qua
gate đó. Không ép đổi `defer` thành `accept` chỉ để qua gate; cần chốt chính
sách class chưa giải quyết trước khi làm thí nghiệm cuối.

Ví dụ một dòng **chỉ sau khi người kiểm tra thực sự đồng ý**:
`selected_synset=airplane.n.01`, `decision=accept`, `reviewer=<tên của bạn>`,
`notes=nghĩa chỉ máy bay phù hợp với nhãn đối tượng`.
Đây là ví dụ cách điền, không phải một quyết định review đã thực hiện.

Nếu không chắc, dùng `defer` và để `selected_synset` trống. Không ép chọn cho
đủ coverage. Không dùng relation annotations hoặc kết quả trên val/test để
chọn nghĩa nhằm tăng điểm. Nếu cần ảnh tham chiếu, lấy từ train và ghi lại
ID/cách lấy mẫu; báo cáo rõ phạm vi đã kiểm tra.

Ví dụ cần chú ý: `bike` được gợi ý `bicycle.n.01`, nhưng cũng có thể là xe máy;
`glass` có thể là cốc uống nước hoặc vật liệu kính. `short` được gợi ý
`short.n.01` trong nguồn cũ, song WordNet định nghĩa đó là vị trí bóng chày;
nếu nhãn trong ảnh chỉ quần short thì cần mở rộng candidate (WordNet 3.0 có
`short_pants.n.01`) sau khi xem ví dụ train. Gợi ý sai vẫn hiển thị để nhóm
thấy lý do cần defer, không phải để chấp nhận mặc định.

Có thể chia các khoảng class ID cho thành viên nhưng cần giữ một bản tổng hợp
và người chịu trách nhiệm từng dòng. Chưa cần quyết định hết mọi dòng trong một
lượt. Script luôn để trống quyết định và tên người kiểm tra; Codex không điền
thay người hoặc tự biến gợi ý thành nhãn đã xác nhận.

## Sau khi kiểm tra

Giữ CSV gốc, manifest nguồn và bản quyết định đã điền. Codex sẽ kiểm tra schema,
giúp compile các quyết định thực tế và triển khai đường export reviewed phù hợp.
Đường reviewed chưa hoàn thiện; không dùng `--allow-draft` để tuyên bố đã qua
kiểm duyệt. Các nhãn defer cần được giữ rõ trong mask và thống kê coverage.

Khi nhóm đã điền xong, trước hết kiểm tra bản quyết định mà không ghi file:

```bash
python -m scripts.prepare_wordnet_review compile \
  --mapping data/knowledge/wordnet-vg150/wordnet_mapping.json \
  --review-csv data/knowledge/level1-human-review-v2/review.csv \
  --output configs/knowledge/wordnet_overrides.reviewed-v2.json \
  --dry-run
```

Script kiểm tra class ID/label, candidate, quyết định và tên reviewer cho các
dòng `accept`. Kết quả `accepted_decisions` là số dòng đã thực sự được người
điền chấp nhận, không phải số gợi ý AI. Sau đó Codex có thể giúp kiểm tra
provenance, compile và dựng lại artifact trong thư mục mới.

`short` là trường hợp phải **defer** nếu nghĩa muốn dùng là quần short: candidate
trong mapping nguồn cũ không chứa `short_pants.n.01`. Compiler hiện tại chỉ
chấp nhận candidate đã có trong nguồn; cần mở rộng candidate với provenance
riêng trước khi chọn nghĩa này.
