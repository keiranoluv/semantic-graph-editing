# Kết quả pilot easy/copy — seed 42

Bốn lượt hoàn tất và đã kiểm chứng lại từ prediction lưu trên đĩa. **Copy giúp xuất đúng tên ngoài từ vựng, nhưng chưa tăng F1 thao tác đầy đủ.** Cả bốn mô hình đều đạt 0/67 exact graph và 0/18 exact trên nhóm có đúng một thao tác tham chiếu. Gate vận hành vượt ngưỡng không có nghĩa là mô hình đã giải tốt nhiệm vụ.

| Mô hình | Copy tên | Micro F1 thao tác | Tên thêm đúng / 42 | Add-node đúng đầy đủ / 42 | Exact graph |
|---|---|---:|---:|---:|---:|
| MLP | Không | 9,83% | 10 | 1 | 0/67 |
| MLP | Có | 9,83% | 11 | 1 | 0/67 |
| GNN | Không | 10,10% | 7 | 1 | 0/67 |
| GNN | Có | 10,07% | 9 | 1 | 0/67 |

Tên đúng chỉ là chẩn đoán bỏ qua thuộc tính và ID, không thay thế chỉ số thao tác. Chênh lệch GNN -0,024 điểm phần trăm tương ứng cùng 21 TP nhưng thêm một FP; không có cơ sở kết luận khác biệt đáng kể từ một seed. Không so trực tiếp với compact-v1 vì nhãn, từ vựng và số mẫu validation khác nhau.

MLP chọn epoch 16, dừng 46; GNN chọn epoch 10, dừng 40, đúng early stopping patience 30. Cả bốn dùng chung 300 train, 67 validation, 219 thao tác tham chiếu. `replace_edge` không có support ở validation; F1 loại này không xác định, không phải 0. Báo cáo JSON giữ đủ precision/recall/F1 theo sáu loại.

## Những gì cơ chế copy làm được

Cả hai mô hình copy xuất đúng `bicycle`, `gorilla`, `guitar` ở ba mẫu cần tên ngoài từ vựng train. Trước đó chúng xuất `bicycles`, `road`, `bow tie` hoặc `sombrero`. Đây là 3/16 tên OOV tham chiếu, và 3/4 tên OOV có mặt trong ứng viên copy. Lightning vẫn chưa được dự đoán đúng.

Lợi ích bị bù một phần bởi lựa chọn span sai: cả hai đổi `carrots` thành `cow eat`; MLP đổi nhãn chuẩn `SUV` thành `luxury suv`. Trường hợp sau có liên quan ngữ nghĩa nhưng vẫn không khớp cách chuẩn hóa nhãn hiện hành. Candidate extractor hiện chưa phân biệt chắc chắn cụm danh từ với cụm động từ.

## Vì sao tên đúng nhưng thao tác vẫn sai

- `338078:1`: đã thêm tên `gorilla` và quan hệ `on road`, nhưng sinh thuộc tính không được yêu cầu và sửa sai thuộc tính của đường. GNN vẫn sai toàn bộ đồ thị dù quan hệ đúng.
- `338078:3`: đã thêm tên `guitar`, nhưng thuộc tính sai, quan hệ ra `wearing` hoặc `on` thay vì `holding`, đồng thời tư thế gorilla sai.
- `484847:1`: tên laptop và một cạnh `on table` đúng, nhưng mô hình vẫn thêm màu/ngoại hình dù schema mới để trống; cũng bỏ sót sách/remote bị xóa trong ảnh đích. GNN thêm một cạnh thừa.
- `357633:1`: nhãn mới giữ đúng `car image` trên screen. MLP không sửa; GNN sửa màu của laptop/screen, không tạo đúng nội dung. Mẫu không còn được tính exact nhờ nhãn bị rỗng như compact cũ.

Có 9 đỉnh thêm ở validation được giản lược về thuộc tính rỗng. Cả hai mô hình copy chọn đúng tên nhưng thêm thuộc tính thừa ở 4 trường hợp trong nhóm này. Hơn nữa, `replace_node` đúng đầy đủ là **0/45** ở cả bốn mô hình. GNN copy chọn đúng loại thao tác và ID ở 22/45, đúng thêm tên đỉnh ở 17/45, nhưng vẫn không đúng toàn payload.

## Giới hạn biểu diễn cần sửa trước khi chạy tiếp

**39/45 thao tác thay đỉnh cần ít nhất một thuộc tính không có trong đầu ra hiện tại. Chỉ 6/45 biểu diễn được đầy đủ, kể cả khi cho copy tên.** Ví dụ các cụm `purple shoulder strap`, `side part`, `blue sides`, `white front`, `car image` chưa thể được sinh bởi đầu thuộc tính đóng. Copy tên đối tượng không sửa được vấn đề này.

Việc khôi phục nhãn gốc giúp giữ ý nghĩa, nhưng cũng khôi phục nhiều cụm thuộc tính chi tiết. Bản “dễ” chỉ giảm ngoại hình của một số vật thể mới, chưa làm đơn giản đủ phần thay thuộc tính. Quyết định thu hẹp đầu thuộc tính còn 429 giá trị cũng làm giảm khả năng biểu diễn: nếu giữ toàn bộ thuộc tính xuất hiện trong train, độ phủ thao tác thay đỉnh sẽ là **13/45**, so với 6/45 hiện tại. Đây là tác dụng phụ của thiết kế vừa triển khai, không phải kết luận rằng chỉ cần train lâu hơn.

13/45 vẫn thấp: chỉ khôi phục từ vựng lớn cũng chưa giải quyết vấn đề. Cần một biểu diễn thuộc tính ngắn, có cấu trúc và giữ đúng thông tin chỉnh sửa — chẳng hạn tách loại thuộc tính và giá trị cho màu/vật liệu/trạng thái/nội dung, với nội dung như `content=car` thay vì buộc phân loại cả cụm tự do `car image`. Đây là hướng cần thiết kế và kiểm chứng từ train; chưa triển khai hay đánh giá phiên bản đó trong báo cáo này.

Loss train tiếp tục giảm nhưng loss validation tăng ở cuối lượt (copy MLP: khoảng 0,70→0,27 train và 1,57→2,04 validation; copy GNN: 0,90→0,29 và 1,52→2,15). Kết hợp với giới hạn từ vựng, chưa có lý do ưu tiên tăng epoch hoặc nhân thêm seed.

## Chẩn đoán CPU về tính nhất quán với schema

Đã thử áp dụng **chính quy tắc giản lược đã khóa** lên đồ thị dự đoán lưu sẵn. Phép biến đổi chỉ dùng nguồn, câu yêu cầu và đồ thị dự đoán; không dùng target, nhãn đúng hoặc số vật thể trong target để quyết định. Nó chỉ bỏ ngoại hình của đỉnh mới khi quy tắc cho phép, giữ các thuộc tính được yêu cầu và trường hợp trùng tên cần phân biệt ID. Nhãn tham chiếu, đủ 67 mẫu và cách chấm không thay đổi.

| Mô hình | F1 sau phép chiếu chẩn đoán | Exact graph |
|---|---:|---:|
| MLP không copy | 10,32% | 0/67 |
| MLP copy | 11,79% | 0/67 |
| GNN không copy | 11,06% | 0/67 |
| GNN copy | 11,99% | 0/67 |

Đây là chẩn đoán hậu xử lý CPU, không phải kết quả chính của launcher hoặc mô hình huấn luyện mới. Các prediction gốc không bị sửa. Nó cho thấy decoder cần tuân thủ schema ở bước xuất nhãn, nhưng sửa điều đó một mình vẫn chưa giải quyết đúng đồ thị.

Tỷ lệ hợp lệ cấu trúc sau ràng buộc là 100%; trước ràng buộc chỉ 70,15% ở hai MLP, 61,19% ở GNN không copy và 58,21% ở GNN copy. Hợp lệ cấu trúc không đồng nghĩa làm đúng yêu cầu.

## Bước tiếp theo và phạm vi

Giữ nguyên nhãn, checkpoint và kết quả lượt này làm đối chứng. Trước GPU tiếp theo, ưu tiên thiết kế lại biểu diễn thuộc tính và kiểm tra độ phủ trên CPU, giữ các thông tin sửa trực tiếp như nội dung màn hình; đồng thời đưa ràng buộc schema vào decoder một cách rõ ràng. Kiểm tra vài trường hợp train bằng memorization trước khi chạy full. Không tiếp tục bỏ thông tin quan trọng, thay ảnh đích bằng ý tưởng từ prompt, hoặc nới cách chấm để nâng điểm.

Task B nhiệm vụ 2 vẫn là dự đoán sáu thao tác từ đồ thị nguồn và câu yêu cầu đóng băng. Thực nghiệm chính ba seed và test chưa được mở rộng trong lượt kiểm tra này. Không có GPU/inference mô hình mới được chạy bởi audit.

Tái lập báo cáo:

```bash
myenv/bin/python -m tools.analyze_magicbrush_easy_copy
```

`result-audit.json` kiểm chứng release, hash cache, bốn completion receipt, ID/thứ tự toàn bộ prediction; chấm lại độc lập cả raw/constrained và tính lại chẩn đoán. Các tệp `*-schema-projection.json` chỉ là prediction của phép chiếu chẩn đoán riêng. Báo cáo này không yêu cầu đổi source/manifest/config đã khóa.
