# Nhãn MagicBrush gọn hơn cho nhiệm vụ 2

Bản compact v1 giảm chi tiết trong thuộc tính của graph, giữ đối tượng, quan hệ và sáu loại thao tác của nhiệm vụ 2. Đây là phiên bản nhãn riêng để phát triển thực nghiệm tiếp theo; v3 và toàn bộ nhãn gốc vẫn được giữ nguyên. Chưa có kết quả huấn luyện trên nhãn compact.

## Kết quả chuyển đổi

| Đại lượng | Gốc | Compact |
| --- | ---: | ---: |
| Mẫu train | 301 | 301 |
| Mẫu validation | 67 | 67 |
| Từ vựng thuộc tính train | 1.448 | 298 |
| Thuộc tính trung bình mỗi node trên train | 1,67 | 1,10 |
| Thao tác train | 765 | 741 |
| Mẫu train có thao tác | 301 | 301 |
| Mẫu validation có thao tác | 67 | 64 |

Từ vựng thuộc tính train giảm **79,4%**. Sáu loại thao tác vẫn có trong train: add_node 159, delete_node 73, replace_node 209, add_edge 209, delete_edge 85, replace_edge 6. Có 37 replace_node biến mất trên train/validation vì khác biệt thuộc tính được loại hoặc quy về cùng khái niệm. Các thao tác thêm/xóa đối tượng và thao tác cạnh được giữ nguyên.

## Quy ước nhãn

- Giữ màu cơ bản, chất liệu, hình dạng, kích thước, hoa văn và trạng thái ngắn, rõ. Ví dụ `wooden → wood`, `dark blue → blue`, `white and pale green → green + white`.
- Giữ thuộc tính gắn đúng bộ phận: `blue shirt → shirt blue`, `white icing → icing white`. Không suy diễn rằng cả người màu xanh hoặc cả vật thể màu trắng.
- Giữ ý chỉnh sửa đặc thù khi có bằng chứng trong annotation: tóc, râu, miệng/mắt, cửa, màn hình, chữ/số. Ví dụ `blond mohawk → hair blond + hair mohawk`, `CK lettering → text CK`, `number 18 → number 18`. Chữ/số được giữ nguyên nội dung vì có thể chính là yêu cầu chỉnh sửa.
- Chỉ giữ vị trí thô `left/right/center/upper/lower`, để phân biệt nhiều vật thể cùng loại. Bỏ mô tả che khuất, cắt khung, độ chi tiết bố cục và trang trí ngoài quy tắc.
- Không giới hạn cứng số thuộc tính bằng cách cắt bớt tùy ý. Các thuộc tính còn lại phải qua quy tắc rõ ràng; thuộc tính không được ánh xạ nằm trong file phụ với nguyên văn và lý do.
- Áp dụng **cùng một hàm độc lập cho từng graph**. Hàm không đọc instruction hoặc graph đối diện, không biết dự đoán mô hình. Sau đó tính lại thao tác từ hai graph mới. Không lấy câu yêu cầu để tự bịa ra chi tiết trong ảnh đích.
- Giữ nhãn đối tượng, ID, cạnh, câu yêu cầu, nhóm ảnh và split. Không gộp nhiều instance thành một vật thể. Việc chuẩn hóa tên đối tượng/quan hệ chưa thuộc phiên bản này.

Các ánh xạ được thiết kế từ annotation train và các nhóm khái niệm đã nêu. Policy và code được khóa trước khi chuyển đổi validation. Không thay quy tắc sau khi xem ba trường hợp validation dưới đây.

## Phạm vi chưa biểu diễn được

| Mẫu validation | Yêu cầu | Chi tiết bị mất |
| --- | --- | --- |
| 453731:1 | Chó nhìn về phía trước | Hướng nhìn và tư thế đầu |
| 456569:3 | Cây bao quanh sở thú | Khác biệt mật độ/bố trí cây |
| 503943:1 | Laptop có trang web màu đỏ | Nội dung và màu từng vùng trên màn hình |

Các mẫu này vẫn ở dataset, được gắn `no_core_edit`; không được coi việc dự đoán rỗng là đã thực hiện đúng yêu cầu ảnh. Khi xây evaluator compact, phải báo cáo riêng độ bao phủ schema (64/67 validation có thao tác), exact trên tập có thao tác và số mẫu không biểu diễn được; không âm thầm bỏ ba mẫu rồi gọi đó là kết quả trên toàn bộ split cũ.

Giảm chi tiết cũng không loại bỏ thay đổi phụ vốn có trong ảnh/annotation: ví dụ yêu cầu thêm đồng hồ nhưng nhãn gốc còn xóa thanh gỗ. Compact giữ nguyên các thay đổi cấu trúc này. Mẫu yêu cầu đổi coffee thành tea vẫn thiếu nhãn `tea` ngay ở bản gốc; phép rút gọn không tự bổ sung. Vì vậy độ bao phủ và tính nhất quán tốt hơn chưa đủ để kết luận bài toán đã được giải quyết.

Từ vựng tên vật thể/quan hệ vẫn là 682/81 trên train. Kiểm tra coverage của nhãn compact trên validation là 180/206 payload thao tác biểu diễn được bằng từ vựng train. Đây là thống kê schema/từ vựng, không phải accuracy hoặc trần chính xác của mô hình; ba mẫu rỗng cần được đọc cùng thống kê này.

## Dữ liệu và tái lập

```bash
myenv/bin/python -m tools.magicbrush_compact_v1.build
```

Lệnh chỉ chạy CPU, kiểm hash rồi xuất bất biến; chạy lại cho cùng nội dung được phép, ghi đè nội dung khác bị từ chối.

- `data/processed/level2/magicbrush-compact-v1/trainval-edits.json`: 368 mẫu với graph và thao tác mới.
- `data/processed/level2/magicbrush-compact-v1/detail-sidecar.json`: nhãn gốc, từng thuộc tính được giữ/đổi/bỏ và các thao tác bị mất. Có thể đối chiếu đầy đủ.
- `configs/level2/magicbrush-compact-v1-policy.json`: quy tắc ánh xạ.
- `impact.json`, `case-impact.json`: thống kê và ảnh hưởng từng mẫu.
- `train-attribute-mapping.json`: toàn bộ 1.448 thuộc tính train và ánh xạ của chúng.
- `review-20-train.json`, `projection-review.json`: 20 mẫu train chọn ngẫu nhiên cố định và nhận xét của assistant về phép chuyển đổi.
- `manifest.json`, `verification.json`: hash artifact và kết quả kiểm tra.

69 nhãn test **chưa chuyển đổi và chưa đọc trong bước này**. Khi đánh giá sau huấn luyện, phải dùng đúng policy đã khóa, báo cáo trường hợp ngoài schema và không sửa từ điển theo test. Split toàn bộ 437 mẫu gốc không thay đổi.

## Đối chiếu đề bài và bước tiếp theo

`Project01_Graph_Editing.pdf`, trang 3–4, yêu cầu ≥300 cặp, ≥20 mẫu kiểm tra tay, dự đoán thao tác từ graph nguồn và nhúng yêu cầu đóng băng, precision/recall/F1 theo loại và ít nhất ba seed cho thực nghiệm chính. Train compact có 301 cặp không rỗng, giữ đủ sáu loại. Xác nhận tay 20/20 trước đây thuộc nhãn gốc; nhận xét của assistant về chuyển đổi không phải xác nhận mới của người dùng và không chứng minh độ đúng thị giác của toàn bộ nhãn compact.

Bước tiếp theo là chuẩn bị adapter/thực nghiệm riêng cho nhãn compact, có báo cáo schema coverage và tái chạy phép thử học thuộc trên cohort train cố định trước khi chạy thực nghiệm chính. MLP/GNN phải dùng cùng nhãn, split và metric compact. F1 compact không được so trực tiếp với F1 v3 đầy đủ để tuyên bố mô hình tốt hơn, vì đáp án và số thao tác đã thay đổi. Không cần tác vụ GPU cho bước chuẩn hóa hiện tại.
