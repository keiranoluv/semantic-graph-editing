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

Để tải **5 ảnh train cho mỗi class** mà không tải cả hai archive nhiều GB,
chạy:

```bash
python -m scripts.download_level1_review_images
```

Ảnh JPEG được lưu theo `image_id` trong `data/raw/vg150/images/` và bảng
`data/knowledge/level1-human-review-v2/review_images.csv` ghi rõ
`class_id → image_id → local_path`, hộp bao và SHA-256. Script lấy đủ 750
cặp class–ảnh trong bảng ví dụ; vì nhiều class dùng chung ảnh, chỉ có 237
JPEG gốc khác nhau. Ảnh được đọc từng file bằng HTTP Range từ
[bản lưu trữ ZIP Visual Genome trên Hugging Face](https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/tree/main)
khi host ảnh gốc không truy cập được. Chạy lại sẽ bỏ qua JPEG đã có; dữ liệu
ảnh và bảng manifest nằm dưới `data/` nên không đưa vào Git.

Để xem ảnh theo tên class mà không phải tra CSV mỗi lần, tạo bản sao có tên
`class_id_label_image_id.jpg`:

```bash
python -m scripts.prepare_level1_review_images
```

Thư mục kết quả là `data/raw/vg150/images_review/`, ví dụ
`001_airplane_2349975.jpg`, `028_chair_712978.jpg`. Class ID được thêm số 0
phía trước để sắp xếp đúng; số cuối là `image_id` gốc, không phải thứ tự mẫu.
Ảnh gốc vẫn giữ nguyên. Khi cần xem chính xác hộp bao của vật thể trong ảnh,
tra lại `review_images.csv` theo `class_id` hoặc `image_id`. Cùng lệnh đó tạo
`data/raw/vg150/images_review/index.html`: mở file này bằng trình duyệt, gõ
`bike` hoặc class ID vào ô lọc; khung đỏ chỉ vật thể được gán nhãn. Đây là
cách nhanh hơn để phân biệt nhiều vật thể trong cùng ảnh.

Mỗi class có 5 file tên riêng trong `images_review/`; cùng một JPEG gốc có
thể được sao chép dưới các tên class khác nhau với **hộp bao khác nhau**.
Chẳng hạn ảnh `2344860` có
chiếc xe máy đỏ ở giữa, với hai hộp gần trùng nhau mang nhãn `bike` và
`motorcycle`. Đây là bằng chứng nhãn `bike` có thể chứa nghĩa xe máy trong
train split, không phải quyết định review toàn bộ class.

Class `building` có hai ví dụ cần chú ý: ảnh `1592671` chụp nội thất nhưng
box `building` phủ khoảng 99% ảnh; ảnh `2389149` có box rất mỏng, chỉ khoảng
0,5% diện tích ảnh. Cả hai box khớp annotation HDF5 gốc sau khi đổi tọa độ,
nhưng nội dung ảnh khó dùng để xác nhận nghĩa của class. Reviewer nên ghi rõ
ví dụ nào bị nhiễu hoặc không đủ thông tin, không xem mỗi box như bằng chứng
ngữ nghĩa chính xác.

Class `cap` cũng đa nghĩa trong 5 ví dụ train: `2347962` trỏ vào mũ len,
`2354978` vào mũ/visor của người chơi tennis, còn `2399043` trỏ vào nắp chai.
Ảnh `2388888` có box trên một chi tiết trắng nhỏ ở thiết bị vệ sinh, chưa đủ
rõ để gọi tên chính xác. Cả 5 box đều khớp HDF5 gốc; vấn đề là một class có
nhiều nghĩa vật thể. Không nên `accept` một synset duy nhất cho cả class chỉ
dựa vào các ví dụ này; ghi `defer` và nêu các image ID/nhóm nghĩa trong `notes`.

Class `ear` có 5 box khớp HDF5 gốc và đều chỉ **tai ngoài nhìn thấy**:
`2348882` (chó), `2360052` (người), `2383751` (bò), `2402376` (cừu),
`2407625` (ngựa vằn). Gợi ý `ear.n.01` là cơ quan nghe nói chung; candidate
`auricle.n.02` chỉ cấu trúc tai ngoài và có thể sát đối tượng trong ảnh hơn.
Không có ví dụ bắp ngô (`ear.n.05`) trong 5 ảnh này. Reviewer cần quyết định
mức cụ thể mong muốn của mapping và ghi lý do; Codex chưa điền `accept` thay.

Class `glass` có 5 box khớp HDF5 nhưng trộn nghĩa: `2362080` và `2398917`
chỉ ly uống, `2396412` và `2407062` chỉ kính đeo mắt. Ảnh `2388478` có box
trên một vùng kính nhỏ của công trình, cần nhìn kỹ trước khi gọi tên. Candidate
nguồn hiện có `glass.n.02` (ly) và `glass.n.01` (chất liệu kính), nhưng thiếu
`spectacles.n.01` (kính đeo mắt), vì bộ sinh candidate dùng từ đơn `glass`.
Thêm candidate này vẫn chưa giải quyết việc **một class chứa nhiều loại vật**;
giữ `defer` và ghi các image ID/nhóm nghĩa trong `notes` trước khi chốt chính
sách mapping hoặc xử lý class đa nghĩa.

Class `wheel` (ID 144) có 5 box khớp HDF5 và đều chỉ **bánh xe**, không bao cả
chiếc xe: `2344544` là bánh xe đạp ở mép phải, `2378325` là bánh xe máy,
`2396132` và `2406051` là bánh xe đạp, `2410124` là bánh toa tàu. Gợi ý
`wheel.n.01` phù hợp với vật thể trong 5 ảnh; candidate `bicycle.n.01` chỉ
cả chiếc xe đạp nên không phù hợp với box. Reviewer có thể xem các ảnh này
để quyết định `accept` và ghi bằng chứng vào `notes`.

Host JPEG gốc trả lỗi kết nối trên máy này. 237 ảnh đã được tải qua bản ZIP
thay thế và kiểm tra mở ảnh, SHA-256, kích thước khớp metadata, hộp nằm trong
ảnh. Người review vẫn cần **tự xem nội dung**; các kiểm tra kỹ thuật không xác
nhận nghĩa WordNet. Nếu cần xem thêm ví dụ ngoài 5 ảnh mỗi class, dùng các URL
trong CSV hoặc archive ảnh tại
[Visual Genome](https://visualgenome.org/api/v0/api_home), tra bằng `image_id`.

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

Giữ CSV gốc, manifest nguồn và bản quyết định đã điền. Bản lựa chọn **150/150**
do Toan cung cấp ngày 2026-10-25 đã được lưu tại
`configs/knowledge/level1_verified_synsets_toan_2026-10-25.txt` (đúng thứ tự
class ID 1..150). Ba dòng trống trong bản paste đầu đã được Toan bổ sung:
`bike=motorcycle.n.01`, `building=building.n.01`, `cap=cap.n.01`. Toan cũng sửa
`orange=orange.n.01`, `stand=base.n.08`, và chọn `glass.n.02` làm nghĩa đại diện
dù class chứa cả kính đeo mắt. Một synset đại diện cho class **không đảm bảo
đúng cho mọi object instance**; xem ghi chú trong override.

Lệnh kiểm tra/biên dịch lại từ file đã lưu:

```bash
python -m scripts.compile_level1_verified_review \
  --reviewer Toan --review-date 2026-10-25 --dry-run
```

Lệnh đã chạy và tạo `configs/knowledge/wordnet_overrides.reviewed-toan-v1.json`
với 150 quyết định `accept`, tên reviewer và ghi chú nguồn; bản CSV đã điền,
mapping mở rộng và report nằm trong
`data/knowledge/level1-human-review-v2/compiled-toan-v1/`. Bản paste chỉ có
cột synset, không có lý do riêng cho hầu hết dòng; ghi chú nói rõ giới hạn này
và tách phần kiểm tra ảnh của Codex khỏi lựa chọn của Toan. Candidate bổ sung
`short_pants.n.01` được xác thực bằng WordNet 3.0 và ghi provenance ở
`configs/knowledge/level1_review_candidate_additions.json`.

Tạo lại mapping WordNet 150/150 từ override đã biên dịch:

```bash
python -m scripts.prepare_knowledge \
  --overrides configs/knowledge/wordnet_overrides.reviewed-toan-v1.json \
  --output-dir data/knowledge/wordnet-reviewed-toan-v1
```

Lệnh này đã chạy: 150 `linked_manual`, 0 `unmatched`. Đây mới là **mapping**;
KG/TransE checkpoint và bảng nhúng C.2 hiện có vẫn thuộc draft 124 class.
Cần dựng và huấn luyện lại theo nguồn reviewed trước khi báo cáo kết quả
GAT+KG reviewed. Bản paste không có `evidence_reference` và lý do riêng cho
từng class như manifest của chế độ TransE `reviewed` yêu cầu; compiler 150
`accept` không tự chứng minh audit từng ảnh. Nhóm cần bổ sung bằng chứng và
rà các class đa nghĩa trước khi mở strict training gate. Không dùng
`--allow-draft` để tuyên bố đã qua kiểm duyệt.
