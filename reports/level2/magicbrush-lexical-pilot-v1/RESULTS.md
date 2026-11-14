# Kết quả pilot liên kết câu yêu cầu với đối tượng

Đã chạy xong MLP và GNN trên RTX 5090 theo quyền người dùng cho phép, sau kiểm tra CPU và commit `cb1ef02`. Cải tiến giúp chọn tên đối tượng tốt hơn nhưng mức tăng F1 thao tác còn nhỏ. Chưa chạy thêm seed hay test; GPU đã dừng.

## So sánh cùng điều kiện

Cùng 301 train, 64 validation có thao tác (báo cáo đầy đủ 67), nhãn compact, cache CLIP, từ vựng, seed 42, optimizer, lịch huấn luyện, decoder và tiêu chí chọn checkpoint. Model mới thêm 72 tham số cùng prior khớp từ mềm đã công bố.

| Model | F1 baseline | F1 mới | Thay đổi | Exact compact | Epoch chọn / dừng |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 11,14% | 13,33% | +2,19 điểm % | 0/64 | 18 / 48 |
| GNN | 10,38% | 10,90% | +0,52 điểm % | 1/64* | 16 / 46 |

Đây là metric của output có ràng buộc. F1 output thô của MLP chỉ tăng từ 10,72% lên 10,83%; GNN tăng từ 8,70% lên 10,90%. Không diễn giải toàn bộ mức tăng của MLP là do nội dung output thô tốt hơn. Cả hai dừng vì 30 epoch không cải thiện tiêu chí chọn checkpoint.

*Exact GNN cần được đọc cùng hạn chế schema bên dưới. Một seed chưa đủ kết luận khác biệt có ý nghĩa thống kê hoặc GNN tốt hơn MLP.

## Tín hiệu mới giúp ở đâu

| Chẩn đoán | MLP trước → sau | GNN trước → sau |
| --- | ---: | ---: |
| Tên vật thể thêm mới khớp, bỏ qua thuộc tính và ID | 2/42 → 8/42 | 2/42 → 8/42 |
| Đúng loại thao tác và ID nguồn, chưa xét payload | 24/105 → 26/105 | 20/105 → 21/105 |
| Thao tác add_node đúng đầy đủ | 1/42 → 1/42 | 0/42 → 0/42 |

Ví dụ `484847:1`, “put a laptop on the table”: cả hai đã chọn `laptop` thay vì `chair`. Nhưng đáp án yêu cầu laptop có `black, open, screen blue`, MLP dự đoán `white`, GNN dự đoán `red`. Đáp án còn có xóa books/remote là thay đổi phụ trong cặp ảnh; model chưa dự đoán các thao tác đó. Chọn đúng tên chưa đủ tạo thao tác đúng đầy đủ.

## Kiểm tra tại checkpoint đã chọn

| Tín hiệu bật khi suy luận | MLP F1 | GNN F1 |
| --- | ---: | ---: |
| Cả nhãn và ID | 13,33% | 10,90% |
| Tắt phần nhãn | 12,78% | 10,90% |
| Tắt phần ID | 8,98% | 10,61% |
| Tắt cả hai | 8,38% | 10,61% |

Khi tắt phần nhãn, tên vật thể thêm đúng giảm từ 8 xuống 1 ở MLP và xuống 2 ở GNN. Tuy vậy, GNN vẫn có add_node đúng đầy đủ bằng 0 nên F1 thao tác không tăng nhờ chọn tên tốt hơn. Đây là ablation **chỉ lúc suy luận trên cùng trọng số đã học**, không phải các model được huấn luyện độc lập.

## Giới hạn còn lại

1. **Từ vựng đóng:** 16/42 tên đối tượng cần thêm nằm ngoài từ vựng train. `gorilla` và `guitar` không có đầu ra tương ứng, nên phiên bản này không thể phát sinh đúng chúng. Chỉ 10/42 tên vừa nằm trong train vừa khớp nguyên cụm từ sau chuẩn hóa với câu yêu cầu; đây là độ bao phủ đặc trưng exact, không phải trần của toàn bộ model.
2. **Vai trò từ trong câu:** yêu cầu gorilla băng qua đường vẫn dẫn đến dự đoán `road`. Khớp tên chưa phân biệt chắc chắn đối tượng cần thêm và đối tượng làm mốc quan hệ; dạng từ khóa như `can` còn có thể bị nhầm với tên vật thể.
3. **Nội dung thuộc tính và thao tác phụ:** số thao tác thêm node đúng đầy đủ chưa tăng. Không nên tiếp tục xóa thuộc tính chỉ để biến các dự đoán hiện tại thành đúng.
4. **Exact có thể che mất ý yêu cầu sau projection:** mẫu GNN đúng duy nhất `357633:1` yêu cầu hình xe trên màn hình laptop. Nhãn gốc đổi `application windows → car image`; compact đổi `screen application → []`, do `car image` bị bỏ. Model chỉ khớp đáp án compact, chưa chứng minh thêm đúng hình xe. `schema-caveat.json` ghi nguyên văn và hash bằng chứng. Ba mẫu validation `no_core_edit` trước đây vẫn giữ nguyên; trường hợp này cho thấy còn có mất thông tin một phần trong mẫu vẫn có thao tác.

## Hướng tiếp theo

Giữ phiên bản này làm đối chứng. Nếu tiếp tục cải tiến mô hình, ưu tiên sao chép tên mới từ chính câu yêu cầu và phân biệt vai trò đối tượng trong câu, với ứng viên chỉ từ input và train. Đồng thời cần kiểm tra quy tắc compact có bảo toàn ý chỉnh sửa trước khi dùng exact như bằng chứng hoàn thành yêu cầu. Mọi sửa schema cần phiên bản riêng và áp dụng nhất quán, không sửa riêng validation hoặc test theo kết quả.

Chưa nên nhân ba seed hoặc mở test chỉ vì gate vận hành đã qua. Bài toán vẫn là graph nguồn + nhúng yêu cầu đóng băng → sáu loại thao tác, đúng phạm vi nhiệm vụ 2; không cần thêm hệ sinh ảnh hoặc giao diện mới cho bước này.

## Chứng cứ và tái lập phân tích

```bash
myenv/bin/python -m tools.analyze_magicbrush_lexical_pilot
```

`result-audit.json` kiểm hash và chấm lại dự đoán chính của cả baseline lẫn phiên bản mới; `schema-caveat.json` ghi vấn đề mất thông tin. Metric ablation được lưu trong lần suy luận bằng code đã commit và kiểm hash; script audit không chạy GPU lại. Các dự đoán/checkpoint đầy đủ nằm tại `runs/level2/magicbrush-lexical-pilot-v1/`.
