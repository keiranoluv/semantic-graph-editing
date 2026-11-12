# GQA WordNet v2 — review, audit train và KG

Đã áp dụng bản review đã chốt: **23 accept, 7 defer**. V2 thêm 8 liên kết,
đổi `food.n.01 → food.n.02`, `wire.n.02 → wire.n.01`, xác nhận lại 13 liên kết.
Bảy nhãn defer giữ synset rỗng và nhận vector KG bằng 0. Những dòng ngoài mẫu
30 nhãn giữ nguyên quyết định v1.

| Audit trên toàn bộ train | V1 | V2 |
| --- | ---: | ---: |
| Lớp có liên kết / 1.699 lớp | 693 | **701 (41,26%)** |
| Object có liên kết / 1.148.828 object | 741.673 | **743.205** |
| Độ phủ theo object | 64,559% | **64,692%** |

Audit đọc đủ **69.942 ảnh train / 274 shard**, kiểm hash và ID; không mở shard
validation/test hoặc annotation JSON. Tần suất dùng để ưu tiên mapping chỉ
lấy từ train. [Coverage](coverage.json), [tần suất từng lớp](object-frequency.csv),
[danh sách chưa liên kết theo tần suất](unlinked-priority.json),
[nguồn đã đọc](train-read-provenance.json).

Những nhãn chưa liên kết thường gặp nhất: wall (20.220 object), ground
(17.112), grass (13.306), leaves (9.998), trees (9.969), pants (8.867), water
(8.184), floor (6.851), clouds (5.858), windows (5.542). Không tự chọn nghĩa
cho các nhãn này; chúng tiếp tục dùng nhánh thị giác/hình học và embedding
nhãn của GAT, với phần KG bằng 0.

## KG và embedding mới

KG v2 lấy từ WordNet 3.0, mở rộng hai hop qua IsA, InstanceOf, HasPart,
HasMember, HasSubstance: **2.103 entity / 2.495 triple**. Xây KG là truy vấn
từ điển và xử lý cấu trúc; không chạy mô hình suy luận. Quan hệ GT của GQA
không được đưa vào KG.

TransE train trên GPU, seed 42, 64D, batch 512, Adam lr 0,001, margin 1;
tối đa 100 epoch, patience 15. Dừng sau 92 epoch; chọn **epoch 77** bằng
validation filtered MRR của split triple WordNet, **MRR 0,018899**. Đây là
metric của KG, không phải mR của PredCls. Held-out triple WordNet được chấm
sau chọn checkpoint; test GQA chưa dùng.

[Config TransE](embedding-manifest.json), [metrics/history](embedding-metrics.json),
[feature bundle manifest](feature-manifest.json). Bảng xuất **[1.700, 64]** có
đủ 701 vector hợp lệ; cả bảy defer và background đều được che. Export kiểm
vocabulary, class/entity ID, hash KG/checkpoint và sự xuất hiện của entity ở
triple train để tránh dùng vector ngẫu nhiên chưa được học.

Review này phủ **mẫu phân tầng 30 nhãn**, không phải xác nhận đúng toàn bộ
1.699 nhãn hoặc ước lượng accuracy đại diện vocabulary. Các lựa chọn kế thừa
VG150 và auto-link ngoài mẫu vẫn được ghi đúng phạm vi. Builder giữ nguyên
flag yêu cầu liên kết/duyệt toàn bộ bằng false; trainer TransE cũ dùng mode
`draft`. Bundle v2 khai báo riêng `gqa_wordnet_v2_partial_human_sample_review`
và giữ provenance của 23 accept/7 defer, không giả mạo hoàn thành review toàn bộ.

## Kiểm kỹ thuật và thực nghiệm chính

[Pilot v2](../gqa-wordnet-pilot-v2/README.md) đã PASS về cùng sampling,
khởi tạo KG-on/off và exact resume trên dữ liệu thật. Full GAT+KG ba seed đã
được khởi chạy tuần tự; [báo cáo vận hành](../gqa-wordnet-full-v2/README.md)
ghi nguồn trạng thái thực tế và W&B. Chưa có bảng kết quả đầy đủ ba seed tại
mốc chuẩn bị này. Artifact v1, 15 lượt full cũ và notebook cá nhân được giữ nguyên.

[Cách tái tạo và bàn giao chạy GPU](../../../docs/LEVEL2_GQA_WORDNET_V2.md).
