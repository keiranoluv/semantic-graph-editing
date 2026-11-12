# WordNet cho GQA — mapping và embedding pilot

Đã chuẩn bị mapping riêng cho **1.699 lớp object GQA**, xây KG từ WordNet 3.0,
train TransE mới và xuất bảng đặc trưng **[1.700, 64]**. Đây là artifact
**draft phục vụ pilot**; chưa có kiểm tra tay mới trong ngữ cảnh ảnh GQA.

| Trạng thái mapping | Số lớp |
| --- | ---: |
| Một nghĩa danh từ, liên kết tự động | 558 |
| Trùng tên chính xác với lựa chọn VG150 đã được Toàn duyệt | 135 |
| Đa nghĩa, chưa chọn synset | 839 |
| Chưa tìm được nghĩa danh từ | 167 |
| Đã liên kết | **693 / 1.699 — 40,79%** |
| Có ít nhất một ứng viên | **1.532 / 1.699 — 90,17%** |

Chỉ chuyển **quyết định theo tên nhãn** từ VG150 sau khi kiểm đủ provenance
của review nguồn. ID lớp được lấy lại từ vocabulary GQA. Toàn bộ embedding
được train mới trên KG GQA. Không tự truyền quyết định từ số ít sang số nhiều.
`linked_manual` trong mapping này chỉ 135 lựa chọn kế thừa; không có nghĩa
chúng đã được kiểm lại trên ảnh GQA.

Nhãn đa nghĩa/chưa khớp nhận vector 0 và valid bit false. Entity không xuất hiện
trong triple train của KG cũng bị che như vậy; lần xuất này cả 693 lớp liên kết
đều có entity đã được train. Một nghĩa danh từ duy nhất vẫn có thể sai nghĩa
thị giác, nên chưa xem auto-link là nhãn đã được người duyệt xác nhận.

## KG và TransE

- Nguồn: WordNet 3.0, fingerprint corpus và source hashes trong
  [coverage.json](coverage.json).
- 581 synset gốc duy nhất → **2.089 entity / 2.477 triple** sau hai hop.
- Quan hệ: IsA, InstanceOf, HasPart, HasMember, HasSubstance.
- Chỉ đọc vocabulary GQA đã tạo từ train. Annotation quan hệ cảnh không được
  dùng để dựng KG; annotation held-out test GQA không được mở.
- Split triple KG transductive, seed 42, tỷ lệ 80/10/10; negative lọc bằng train,
  validation ranking lọc bằng train + validation.
- TransE 64D, Adam lr 0,001, margin 1, batch 512; tối đa 100 epoch,
  patience 15. Dừng epoch 66, chọn **epoch 51**, validation filtered MRR
  **0,021069** trên đủ 247 triple. [Config](embedding-manifest.json),
  [history/metrics](embedding-metrics.json).
- Held-out **triple KG** được chấm sau chọn checkpoint. Đây là split riêng
  của WordNet, không phải test GQA. MRR này không phải mean Recall của PredCls.

## Kiểm tra mapping

[review_all.csv](review_all.csv) giữ đủ 1.699 dòng, ứng viên/định nghĩa và cột
để người duyệt điền. [review_sample.csv](review_sample.csv) có 30 nhãn lấy mẫu
cố định seed 42, phân tầng theo trạng thái. Các cột `decision`, `selected_synset`,
`reviewer`, `notes` đều để trống; chưa ghi nhận human review mới.

Mẫu phân tầng giúp xem các loại lỗi, không tự cho ước lượng accuracy đại diện
toàn bộ vocabulary. Khi kiểm tay cần đối chiếu ảnh train và công bố số mẫu,
cách chọn, số mapping đúng/sai/chưa đủ bằng chứng. Đây là kiểm tra mapping KG;
phần kiểm tay ≥20 cặp chỉnh sửa MagicBrush của đề là một hạng mục khác.

## Kiểm chứng và bước tiếp

[Pilot GAT/KG-off/KG-on](../gqa-wordnet-pilot-v1/README.md) đã PASS về pipeline,
sampling và exact resume. [verification.json](verification.json) xác nhận 73
file nguồn của thực nghiệm cũ, 15 result/best checkpoint và notebook cá nhân
giữ đúng hash. [feature-manifest.json](feature-manifest.json) lưu alignment và
provenance đầy đủ của bảng xuất.

Tiếp theo: kiểm mapping trên ảnh train, ưu tiên các nhãn đa nghĩa thường gặp,
rồi khóa mức độ phủ/protocol trước khi chạy GAT+KG ba seed trên full train.
Chưa chạy main GAT+KG; chưa dùng test GQA; chưa kết luận lợi ích của tri thức.

Lệnh tái tạo được ghi trong [hướng dẫn GQA KG](../../../docs/LEVEL2_GQA_KG.md).
