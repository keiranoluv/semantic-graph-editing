# Kiểm tra định tính 5 hit / 5 miss — VG150 PredCls có CLIP

Ghi nhận ngày 2026-10-25. Đây là **nhận xét sơ bộ của trợ lý khi xem JPEG và
GT box**, chưa phải xác nhận thủ công của thành viên nhóm. Các ví dụ lấy từ
checkpoint GAT+CLIP seed 42 `best.pt` (epoch 45), chạy suy luận CPU trên 100
ảnh đầu của validation rồi xáo thứ tự ảnh bằng seed 2026. Script chọn 5
triplet GT khớp top 50 và 5 triplet GT không khớp. Mười triplet nằm trong
**5 ảnh**, không đại diện cho tỷ lệ lỗi trên toàn validation hoặc cả 15 run.

Nguồn kiểm chứng:

- [JSON dự đoán, GT, box và rank](clip-v1-gat-seed42-example-candidates-val100.json)
- [Ảnh 5 hit](clip-v1-gat-seed42-review-hits.png) và
  [ảnh 5 miss](clip-v1-gat-seed42-review-misses.png)
- [Report đánh giá checkpoint trên 5.000 ảnh val](clip-v1-gat-seed42-val5000.json)

## Năm hit theo metric R@50

| Ảnh | GT và dự đoán | Rank | Nhận xét sơ bộ trên ảnh/box |
|---|---|---:|---|
| 40 | `flower in vase` | 1 | Hoa và bình được khoanh đúng; quan hệ `in` rõ ràng. Box hoa khá rộng, chứa thêm bối cảnh. |
| 7 | `woman sitting on chair` | 14 | Người đang ngồi trên ghế, phù hợp ảnh và box. |
| 89 | `door on table` | 28 | Hit theo nhãn GT nhưng box “table” trùm chiếc tủ đỏ; cánh cửa có vẻ là **bộ phận của tủ**. Không nên gọi đây là thành công ngữ nghĩa chắc chắn. |
| 81 | `door on cabinet` | 36 | Đúng cặp cánh cửa/tủ; `on` có thể chấp nhận, nhưng quan hệ bộ phận như `of` cũng hợp lý. |
| 3 | `girl with hair` | 5 | Hợp lý trên ảnh, song là quan hệ thuộc tính khá hiển nhiên. |

## Năm miss theo metric R@50

| Ảnh | GT → dự đoán | Rank cặp | Nhận xét sơ bộ trên ảnh/box |
|---|---|---:|---|
| 40 | `table near chair` → `table with chair` | 48 | Bàn ở gần ghế; `with` quá chung chung. Đây là lỗi quan hệ rõ nhất trong mẫu. Điểm cặp chỉ 0,014. |
| 7 | `laptop above desk` → `laptop on desk` | 3 | Laptop nằm trên mặt bàn; `on` hợp lý, có thể tự nhiên hơn `above` trong GT. Miss do khác nhãn chính xác. |
| 89 | `shelf on table` → `shelf under table` | 59 | Box “shelf” ở tầng thấp của bàn nhỏ; `under` có cơ sở thị giác, nhưng cả nhãn vật và quan hệ cần người trong nhóm xác nhận. Khác predicate và cặp nằm dưới ngưỡng top 50. |
| 81 | `door on cabinet` → `door of cabinet` | 29 | `of` diễn đạt quan hệ bộ phận khá hợp lý; GT `on` và dự đoán có thể cùng chấp nhận tùy quy ước gán nhãn. |
| 3 | `girl has hair` → `girl with hair` | 5 | Cùng cặp object với hit `girl with hair`. GT chứa hai predicate cho cùng cặp, còn protocol chỉ lấy top-1 predicate/cặp; `has` và `with` gần nghĩa. Đây là miss exact-match, không phải bằng chứng mô hình không thấy tóc. |

## Cách diễn giải

Theo quan sát sơ bộ, khoảng 3 hit có vẻ rõ và 2 hit cần xem lại ngữ nghĩa;
trong 5 miss, 1 lỗi quan hệ khá rõ còn 4 trường hợp có nhãn hoặc cách diễn đạt
mơ hồ. **Không dùng các con số nhỏ này làm tỷ lệ lỗi của mô hình.** Khi viết
báo cáo, phân biệt rõ: (1) khớp triplet GT theo R@50, (2) box đúng vùng,
(3) quan hệ có hợp lý về nghĩa. Một thành viên nhóm cần mở ảnh/box và xác nhận
từng hàng trước khi gọi đó là phân tích thủ công chính thức.
