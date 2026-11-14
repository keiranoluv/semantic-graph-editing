# Chẩn đoán v3 và hướng đi bám nhiệm vụ 2

Kết luận: giữ v3 làm baseline có kết quả thật. Hai vấn đề đồng thời tồn tại: (1) nhãn có độ chi tiết không nhất quán và nhiều chi tiết ảnh đích không xác định duy nhất từ câu yêu cầu; (2) mô hình chọn sai thực thể và nội dung ngữ nghĩa ngay cả trên các mẫu rõ. Chưa có bằng chứng rằng tăng epoch, đổi GNN lớn hơn hoặc chỉ nới cách chấm sẽ giải quyết được.

## Đối chiếu đề bài

Căn cứ `Project01_Graph_Editing.pdf`, trang 3–5:

| Yêu cầu | Hiện trạng / giới hạn |
| --- | --- |
| ≥300 cặp MagicBrush/InstructPix2Pix | Có 437 cặp được chấp nhận; train 301, validation 67, test 69 |
| Kiểm tra tay ≥20 mẫu và báo cáo độ tin cậy | Có xác nhận người dùng 20/20 mẫu train trước đây; không suy rộng sang tất cả nhãn |
| Graph nguồn + nhúng yêu cầu đóng băng → thêm/xóa/thay đỉnh, cạnh | V3 dùng CLIP đóng băng, MLP/GNN có học, đủ sáu loại thao tác |
| Precision/recall/F1 từng loại, exact match yêu cầu đơn giản | Có bộ chấm cố định; đơn giản được quy ước một thao tác đích, phải ghi rõ đây là proxy |
| Cùng split, ≥3 seed cho thực nghiệm chính | V3 đã hoàn tất sáu lượt, cùng split, không chọn báo cáo riêng seed tốt |
| Phân tích 5 dự đoán đúng và 5 sai, notebook minh họa | Chưa có mẫu validation/test đúng toàn bộ graph; không thể gọi năm thao tác đúng một phần là năm mẫu đúng hoàn toàn |
| Các yêu cầu khác của cấp độ 2 | Hai tập cho dự đoán quan hệ và phương pháp từ 2023 trở lại đây cần được chứng minh riêng; audit MagicBrush này không thay thế chúng |

Đề không bắt sinh JSON ký tự, không quy định ngưỡng F1 5% hay tỷ lệ hợp lệ 50%, không yêu cầu LLM mới hoặc sinh ảnh cho nhiệm vụ 2. Từ vựng hữu hạn là lựa chọn có giới hạn của kiến trúc, không phải tiêu chí của đề. Thay đổi ontology/cách biểu diễn chỉ nên làm thành phiên bản có giải thích và đối chứng; thay đổi lớn cần đối chiếu quy định trao đổi với giảng viên ở trang 6.

## Phạm vi và phương pháp

- Chỉ dùng train/validation; không đọc nhãn test hoặc dùng kết quả test để chọn hướng mới trong script này.
- Định lượng toàn bộ 67 mẫu validation cho cả sáu lượt bằng dự đoán đã lưu, kiểm tra receipt/hash và chấm lại theo bộ chấm chính.
- Chọn ngẫu nhiên 20/67 cặp validation được chấp nhận với seed **20261114**, sau khi sắp ID; 20 cặp thuộc 18 nhóm nguồn. Không lấy mẫu theo lỗi, không lấy từ 97 cặp validation thô có cả mẫu deferred.
- Phân tích định tính dùng **GNN seed 42**, là seed đầu cố định, không chọn theo test. Đã nhìn cả ảnh nguồn và ảnh đích của cả 20 cặp, đối chiếu graph, câu yêu cầu, nhãn thao tác và dự đoán.
- Các nhận định ảnh là phân tích của assistant, không phải một đợt xác nhận tay độc lập của người dùng và không thay thế receipt 20/20 trước đây.
- Không chạy model inference/train/GPU; không sửa dataset, nhãn, cấu hình, code đã khóa hoặc notebook của người dùng.

Tái lập phần định lượng:

```bash
myenv/bin/python -m tools.analyze_magicbrush_v3_errors
```

Các file: `quantitative.json` (sáu lượt và từng mẫu), `cases.json` (packet 20 mẫu), `qualitative.json` (nhận định có mức tin cậy), `CASES.md` (đọc từng trường hợp), `same-image-granularity.json` (bằng chứng cùng ảnh nhưng độ chi tiết graph khác nhau).

## Phát hiện có số liệu

### 1. Chọn thực thể và phạm vi chỉnh sửa còn yếu

Trên 67 validation, có **118 thao tác lên thực thể đã tồn tại trong nguồn**. Với GNN/42:

- Chỉ **37/118** đúng cả loại thao tác và ID thực thể.
- **80/118** không chỉnh sửa thực thể cần sửa; một trường hợp chạm đúng thực thể nhưng chọn sai loại thao tác.
- Có **104 thao tác trên thực thể nguồn không nằm trong tập cần chỉnh sửa của nhãn**. Đây là số thao tác, không phải số mẫu.
- Trong 37 thao tác đúng loại/ID, **19 đúng đầy đủ**, **18 sai nội dung**. Cả 18 trường hợp này là thay đỉnh: 12 giữ đúng nhãn đối tượng, nhưng **0/18 có bộ thuộc tính đúng**.

Kết quả không chỉ xuất hiện ở một seed: số đúng loại/ID của sáu lượt nằm trong **21–37/118**; không lượt nào có thuộc tính chính xác trong những `replace_node` đã chọn đúng ID/loại.

Ví dụ `9772:1`: xóa đúng TV và cạnh của TV, nhưng còn xóa person reflection và các cạnh khác. `124157:1`: câu yêu cầu bỏ khay, mô hình bỏ chuối. Đây là lỗi phạm vi chỉnh sửa/ngữ nghĩa, không thể giải thích chỉ bằng metric khắt khe.

### 2. Thêm đỉnh sai ngay từ tên đối tượng

Có 42 đỉnh mới cần thêm trên validation. So khớp đa tập tên trong từng mẫu, tạm bỏ qua ID/thuộc tính để chẩn đoán, GNN/42 chỉ có **2/42** tên trùng. Sáu lượt chỉ đạt **1–2/42**. Chỉ số này không chứng minh căn chỉnh đúng instance và không thay thế F1 chính.

Ví dụ: yêu cầu thêm child → dự đoán duck; gorilla → duck; laptop → dog. Với `456569:2`, label lion đúng nhưng thuộc tính và một quan hệ sai. Do đó, bỏ thuộc tính khỏi cách chấm cũng không chữa được phần lớn lỗi.

### 3. Giám sát ngữ nghĩa rất thưa

Chỉ thống kê nội dung thực sự cần học trong thao tác train:

- Có **473 giá trị thuộc tính cần thêm**, trong đó **394 giá trị chỉ xuất hiện một lần** (83,3%).
- Có **123 tên đối tượng trong thao tác add_node**, trong đó **100 tên chỉ xuất hiện một lần** (81,3%).

Các số này khác kích thước từ vựng v3 (682 label, 1.448 attribute) vì từ vựng gồm cả nội dung ở graph nguồn. Có một giá trị trong danh sách lớp không có nghĩa mô hình đã có nhiều ví dụ giám sát để học sinh giá trị đó.

Độ bao phủ v3 trên validation: 154/219 payload thao tác biểu diễn được; chỉ 15/67 mẫu đủ toàn bộ payload. Mọi mẫu vẫn nằm trong mẫu số đánh giá.

### 4. Nhãn có thể đúng với ảnh nhưng không xác định duy nhất từ yêu cầu

Trong mẫu audit, đánh giá định tính ghi nhận **15/20** có chi tiết đích không được xác định duy nhất từ câu lệnh; **6/20** có thay đổi thực thể phụ ở ảnh đích; **7/20** có mơ hồ về tham chiếu, che khuất hoặc quy ước annotation. Các nhóm chồng lấp, không phải tỷ lệ nhãn sai của dataset.

Ví dụ:

| Câu yêu cầu | Nội dung nhãn ngoài phần yêu cầu rõ |
| --- | --- |
| Make the hydrant red | Thân nhỏ, nắp tròn mới, dây xích |
| Add a kid looking at the plane | Tóc vàng, áo nâu, tư thế cụ thể |
| The refrigerator should be white | Đổi đồ trên nóc tủ, mất áo ở tiền cảnh, đổi số ngăn; tổng 8 thao tác |
| Take the almonds out of the box | Ảnh/nhãn bỏ cả cốc xanh; mô hình chỉ bỏ hạnh nhân và cạnh chứa nó |
| Put a laptop on the table | Mất sách/remote, máy đen và màn hình xanh |

Những thay đổi này phần lớn nhìn thấy ở ảnh đích, nên không được tự coi nhãn là bịa hoặc xóa hết chúng để nâng điểm. Nhưng chúng làm nhiệm vụ dự đoán từ graph nguồn + câu yêu cầu có nhiều nghiệm hợp lý. Đề cho phép xây graph từ cặp ảnh; không có quy định buộc loại tất cả thay đổi phụ. Cần mô tả rõ graph đang ghi nhận thay đổi quan sát được hay thao tác yêu cầu, và không trộn hai định nghĩa mà không báo cáo.

### 5. Có bằng chứng độ chi tiết annotation thay đổi ngay trên cùng ảnh

Ảnh đích `171050:1` và ảnh nguồn `171050:2` có **cùng file SHA-256 và cùng pixel SHA-256**. Hydrant của lần thứ nhất có `hanging chains, narrow, red, rounded cap`; lần thứ hai chỉ có `red`.

Điều này không chứng minh một trong hai mô tả sai về thị giác. Nó cho thấy thiếu một quy ước ổn định về giữ thuộc tính nào; so khớp payload chính xác trở nên nhạy với phong cách gán nhãn. Trường hợp này là bằng chứng cụ thể để xem xét chuẩn hóa schema, thay vì kết luận chung rằng toàn bộ dataset không đáng tin.

## Hướng tiếp theo, giới hạn trong nhiệm vụ 2

**Ưu tiên 1 — kiểm tra học thuộc 16 mẫu train thật bằng v3 hiện tại.** Chọn tập cố định từ train, có đủ các nhóm thao tác, lưu ID/hash, tắt regularization cho thử nghiệm chẩn đoán và đánh giá lại chính 16 mẫu. Không dùng test, không lấy số điểm học thuộc làm kết quả tổng quát hóa. Nếu vẫn không học được, kiểm tra loss/căn chỉnh vị trí/các đầu dự đoán trước khi đổi dữ liệu hoặc mở rộng mô hình. Nếu học thuộc được nhưng validation yếu, ưu tiên độ nhất quán nhãn và liên kết câu yêu cầu với thực thể. Kiểm tra mã hóa 301 mẫu và học một ví dụ giả lập đã qua, nhưng chưa thay thế phép thử này. Tác vụ GPU này chưa được triển khai hay khởi chạy trong audit; cần launcher đã kiểm tra/commit theo AGENTS trước khi bàn giao.

**Ưu tiên 2 — soạn schema gọn, có quy ước từ train trước khi sửa hàng loạt.** Chốt mức đối tượng/nhóm đối tượng, thuộc tính nào là màu/chất liệu/trạng thái, cách ghi đồng nghĩa và quy tắc hiện diện khi bị che khuất. Giữ nguyên ID và split; tạo bản mới cùng provenance nếu áp dụng. Quy tắc áp dụng nhất quán lên cả graph nguồn và đích. Kiểm tra lại độ bao phủ, số cặp (ít nhất 300) và một mẫu gán nhãn theo schema mới. Không sửa riêng test, không đổi tên đáp án theo dự đoán, không âm thầm bỏ mẫu ngoài từ vựng. Tách ghi chú thay đổi phụ/chi tiết không được chỉ định để phân tích; nếu đổi định nghĩa nhãn chính thì phải công bố thành protocol riêng và giữ kết quả cũ làm đối chứng.

**Ưu tiên 3 — chỉ thay một thành phần sau chẩn đoán.** Nếu khả năng học thuộc đạt, kiểm tra liên kết câu yêu cầu với ID nguồn và tên đối tượng mới trước khi tăng kích thước GNN. Một hướng hẹp để thử là cho đầu dự đoán chọn/sao chép các tên được nêu trong yêu cầu, với candidate chỉ từ đầu vào và train; vẫn học các thao tác đủ sáu loại. Đây mới là giả thuyết, chưa được chứng minh sẽ nâng F1. Chạy một pilot train/validation có điều kiện dừng rồi mới nhân ba seed.

Không mở rộng sang sinh ảnh, mô hình thị giác mới, web/UI hoặc LLM-as-a-judge trong bước này. Không tăng epoch một cách mặc định và không nới metric chính để biến sai thành đúng. Mục tiêu tiếp theo là một phép thử có khả năng phân biệt lỗi học/triển khai với lỗi tổng quát hóa, cùng một schema có thể giải thích được khi vấn đáp.
