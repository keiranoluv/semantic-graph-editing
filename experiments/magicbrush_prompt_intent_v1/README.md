# MagicBrush, thao tác đích theo câu yêu cầu (v1)

Mỗi dòng được giữ chứa đồ thị nguồn `source_graph`, câu MagicBrush gốc `instruction`, danh sách thao tác `operations`, ID nhóm ảnh, SHA-256 hai ảnh và danh sách `graph_diff_operations` để làm đối chứng. Người gán nhãn xem hai ảnh và hai đồ thị của bản xuất MagicBrush v2. Đầu vào của mô hình chỉ gồm đồ thị nguồn và câu yêu cầu. Đồ thị ứng với nhãn theo prompt được tạo bằng cách áp các thao tác vào đồ thị nguồn.

## Quy tắc gán nhãn

- Giữ thay đổi mà prompt yêu cầu và các thao tác cấu trúc cần thiết. Ví dụ với “put a laptop on the table”, thêm đỉnh laptop và cạnh từ laptop đến đúng đỉnh table với quan hệ on.
- Khi xóa đỉnh, ghi rõ các cạnh cần xóa hoặc đổi đầu mút. Khi đổi thuộc tính, dùng replace_node trên cùng ID và giữ các thuộc tính nguồn vẫn phù hợp.
- Bỏ các thay đổi tình cờ của ảnh chỉnh sửa. Khi ảnh và đồ thị chưa đủ để xác định đối tượng, trạng thái hoặc một đáp án cho phép chấm tự động, đưa mẫu vào tệp held kèm lý do. Với riêng mẫu bò ăn cà rốt, dùng cạnh near làm đại diện theo quy ước người dùng đã duyệt.
- Các cặp gán nhãn có hỗ trợ AI. Trường origin ghi các trường hợp kế thừa từ pilot train hoặc 20 mẫu hiệu chỉnh validation. Khối dữ liệu mở rộng được đối chiếu ảnh bởi trợ lý. Phiếu 20 mẫu lấy ngẫu nhiên bên dưới dùng để đo độ tin cậy bằng người duyệt.

## Dữ liệu và kiểm tra

| Split | Đầu vào | Giữ | Chờ phân xử | Nhãn khác graph diff trong số giữ |
| --- | ---: | ---: | ---: | ---: |
| Train | 301 | 240 | 61 | 205 |
| Validation | 67 | 46 | 21 | 43 |
| Test | 69 | 59 | 10 | 48 |

Xem `data/manifest.json` để lấy checksum, nguồn gốc và số thao tác từng loại. Mọi cặp giữ có thể dựng lại đồ thị sau chỉnh sửa. Nhóm COCO và hash pixel của cả hai vai ảnh tách biệt giữa ba split. Có ba thao tác replace_edge trong train, số tham chiếu cho loại này ở validation và test bằng 0.

Từ gốc repo với môi trường myenv và dữ liệu MagicBrush cục bộ, tạo phiếu HTML để mở trực tiếp trong trình duyệt.

```bash
myenv/bin/python -m experiments.magicbrush_prompt_intent_v1.review --html /tmp/magicbrush-review20.html
```

Phiếu dùng 20 ID lấy ngẫu nhiên không hoàn lại từ 345 cặp đã giữ, seed 20261119. Danh sách ở `review20.json`. Nút tải kết quả xuất lựa chọn, ghi chú, ID phiên bản dữ liệu và tỷ lệ số mẫu đạt trên 20 khi đã điền đủ. Tỷ lệ này đo mức người duyệt chấp nhận toàn bộ nhãn của một mẫu. Duyệt 20 mẫu hiệu chỉnh trước đây và kiểm tra hợp lệ bằng code là các nguồn bằng chứng riêng.

Ngày 19/11/2026, người duyệt xác nhận cả 20 mẫu trong phiếu đều đạt. Tỷ lệ chấp nhận toàn bộ danh sách thao tác đích trên mẫu kiểm tra tay là **20/20 (100%)**. Kết quả của từng ID và nguồn xác nhận được lưu trong [phiếu kết quả](review20-result.json).

## Chạy so sánh hai cách gán nhãn

```bash
bash scripts/run_magicbrush_prompt_intent_v1.sh --cpu-check
bash scripts/run_magicbrush_prompt_intent_v1.sh
```

Lệnh đầu kiểm tra trên CPU. Lệnh sau do người dùng chạy trên một GPU với W&B đã đăng nhập, theo dõi online trong project semantic-graph-editing, group magicbrush-prompt-intent-v1. Mỗi seed chạy graph_diff rồi prompt, tuần tự qua 42, 123, 3407. Hai nhánh dùng cùng 240 ID train, 46 ID validation, từ vựng lấy từ hai bản nhãn của cùng tập train, cùng khởi tạo và thứ tự batch cho từng seed. Cả hai chọn checkpoint theo nhãn prompt trên validation. Mỗi lượt có run ID riêng và trường train_policy trong W&B.

Dùng GNN 2 lớp, hidden 192, CLIP text ViT-B/32 đóng băng, batch train 16, batch validation 32, AdamW với lr 0.0003. Tối đa 200 epoch, tối thiểu 30, patience 30 epoch. Kiểm tra validation ở epoch đầu và mỗi 2 epoch. Tái sử dụng cache CLIP đã khóa, các chuỗi bổ sung được mã hóa trên CPU bằng cùng bộ trọng số. Mỗi lượt ghi phiên bản thư viện hiện tại trong protocol riêng.

Kết quả từng lượt nằm ở `runs/level2/magicbrush-prompt-intent-v1/{graph_diff,prompt}/seed-<seed>/`. Chạy lại cùng lệnh để tiếp tục từ checkpoint. Sau sáu lượt, launcher dừng ở validation. Bước đánh giá test và lập bảng precision/recall/F1 từng thao tác thực hiện riêng sau khi đã chọn checkpoint.
