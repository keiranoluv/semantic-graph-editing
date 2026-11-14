# MagicBrush: nhãn bảo toàn nội dung và sao chép tên mới

Phiên bản này sửa hai giới hạn của compact/lexical pilot: mất thông tin chỉnh sửa trong phép rút gọn nhãn và bộ tên đối tượng đầu ra đóng. Đây là pilot phát triển, chưa phải kết quả chính ba seed, chưa đánh giá test.

## Dữ liệu và review ảnh

Dữ liệu xuất riêng tại `data/processed/level2/magicbrush-easy-copy-v1/trainval-edits.json`, kèm bản gốc trong `detail-sidecar.json`. Tệp cũ không bị sửa. Có **300 train + 67 validation**, giữ source-group/split cũ; 69 mẫu test cũ chưa được chiếu sang schema mới và không được mở trong công việc này.

Đã nhìn trực tiếp **72 cặp / 144 ảnh** và đối chiếu nội dung chỉnh sửa với nhãn: 20 train chọn ngẫu nhiên seed 20261115, 7 validation có lỗi đã biết, 45 cặp bổ sung mà chính sách giản lược ban đầu tác động. Hash ảnh, cách chọn và ghi nhận từng mẫu ở `visual-review.json`. Đây là review bằng thị giác của assistant, không phải 72 xác nhận mới của người dùng và không suy rộng thành tỷ lệ tin cậy toàn bộ dataset. Review 20/20 trước đây của người dùng áp dụng nhãn gốc, không tự động chứng nhận mọi nhãn mới.

Mẫu train `magicbrush:279710:1` được cách ly: yêu cầu đổi skateboard thành stilt nhưng ảnh/nhãn chỉ đổi chi tiết giày, không có stilt nhìn thấy. Mẫu vẫn được giữ trong export với `status=quarantined`; loader không dùng để train. Không loại mẫu validation theo độ khó hoặc kết quả dự đoán. Mẫu stilt cũng nằm trong 20 mẫu người dùng đã kiểm tra: nhãn có thể đúng với ảnh nhưng yêu cầu không được thực hiện. Cả 20 payload được giữ nguyên, trong đó 19 còn dùng và 1 cách ly; `human-review-coverage.json` ghi rõ phạm vi này. Không báo thành 20 nhãn mới đã được người dùng xác nhận trong phần dữ liệu được chọn.

Quy tắc mới giữ nguyên đồ thị nguồn và mọi đỉnh đích đã tồn tại. Nó chỉ bỏ ngoại hình không được yêu cầu của đối tượng mới trong câu thêm vật thể đơn giản. Các trường hợp có màu, vật liệu, trạng thái, nội dung hình vẽ/màn hình, hoặc nhiều đối tượng mới trùng tên cần phân biệt ID được bảo vệ. Nhãn, cạnh, số lượng vật thể và các thay đổi cấu trúc phụ vẫn giữ nguyên. Chính sách không tạo đỉnh đích chỉ vì câu yêu cầu nhắc tới nó.

- `put a laptop on the table`: có thể bỏ đen/màn hình xanh/mở của laptop mới; vẫn giữ thao tác xóa sách/remote thực sự xuất hiện trong cặp ảnh.
- `put a car on the screen`: giữ `application windows → car image`, không biến thành xóa thuộc tính hoặc thêm xe vật lý.
- Hướng đầu chó, mật độ cây, trang web đỏ, người đang ngủ, tranh bông hoa đều giữ nội dung chỉnh sửa. Review mở rộng đã phát hiện cần bảo vệ riêng các dạng trạng thái và hình vẽ này trước khi khóa chính sách.
- Cà phê thành trà vẫn chỉ được gán các dấu hiệu nhìn thấy của đồ uống đích, không suy ra chắc chắn trà chỉ từ yêu cầu. Một số ảnh chỉ thực hiện yêu cầu một phần.

Kết quả: **46 cặp, 57 đỉnh mới, 113 giá trị thuộc tính phụ** được giản lược; tất cả 46 cặp đã nằm trong review ảnh. Đây là giảm độ khó có phạm vi hẹp, không phải bộ nhãn tối giản toàn cảnh. So với compact-v1, các thuộc tính từng bị xóa được khôi phục từ nhãn gốc; vì vậy không so trực tiếp F1 giữa hai schema.

Để không bắt mô hình phân loại các chi tiết chỉ xuất hiện trong ngữ cảnh, từ vựng thuộc tính đầu ra chỉ chứa **429 giá trị được thêm bởi thao tác train**, thay vì 1.414 giá trị ngữ cảnh. Thuộc tính nguồn bất kỳ vẫn được giữ/xóa qua cơ chế sao chép nguồn. Không bỏ giá trị thuộc tính nào thực sự cần thêm trong train. Cả bốn nhánh dùng chung quy tắc này.

## Cơ chế sao chép và đối chứng

Input vẫn là **đồ thị nguồn + câu yêu cầu**; ảnh đích và nhãn thao tác chỉ phục vụ gán nhãn/giám sát. Output vẫn là sáu loại thêm/xóa/thay đỉnh-cạnh. MLP và GNN dùng CLIP text đóng băng, không train bộ sinh ảnh.

Mỗi câu yêu cầu tự tạo tối đa 64 ứng viên cụm từ liên tiếp dài 1–4 từ, chuẩn hóa số nhiều và chặn từ chức năng/động từ phổ biến. Không lấy ứng viên từ target hoặc từ điển validation/test. Vector CLIP của các cụm từ chỉ là đặc trưng input, không được thêm vào từ vựng sinh toàn cục.

Đầu chọn tên dùng cùng truy vấn ngữ nghĩa học được để chấm từ vựng train và ứng viên copy. Hai đường có cùng cách viết được gộp xác suất vào một lớp; nhờ đó nhãn train đã có trong từ vựng vẫn huấn luyện đường copy. Tên mới dùng chỉ số riêng của từng mẫu khi giải mã. Các test kiểm tra gradient và thực sự học/phát ra tên ngoài từ vựng cố định, không chỉ kiểm tra có chuỗi tên trong danh sách.

Trong 42 đỉnh cần thêm ở validation, số tên có thể biểu diễn bằng từ vựng hoặc copy tăng **26 → 30**: thêm `bicycle`, `gorilla`, `lightning`, `guitar`. Đây là độ phủ ứng viên, **không phải độ chính xác**. 12 tên còn lại chưa được giải quyết; từ đồng nghĩa, cụm tên diễn đạt khác, thuộc tính/quan hệ ngoài từ vựng và ID nguồn vẫn là giới hạn. Candidate extractor không phải bộ phân tích danh từ hoàn chỉnh.

Thay đổi schema đã tham khảo lỗi train và validation đã quan sát; không gọi đây là thiết kế hoàn toàn chỉ từ train. Không dùng test để chọn quy tắc. So sánh hợp lệ của pilot là **no_copy với copy trên cùng nhãn mới**; chưa tách nhân quả giữa từng thay đổi nhãn và việc thu hẹp từ vựng thuộc tính.

## Kiểm tra và chạy

CPU: 47 test liên quan đã qua, gồm 19 test mới; `cpu-smoke.json` ghi kiểm tra trên 16 mẫu train thật với vector CLIP thật tính trên CPU, một bước optimizer cho mỗi nhánh. Tất cả 367 bộ target mã hóa được; loss/gradient hữu hạn và giải mã được. Đây không phải kết quả thực nghiệm train hoàn chỉnh. Kiểm tra preflight còn xác minh các khóa nguồn cũ, provenance ảnh, tính tái dựng graph và việc giữ đầy đủ validation.

Theo `AGENTS.md`, assistant không tự mở tác vụ GPU. Chạy từ thư mục repo:

```bash
bash scripts/run_magicbrush_easy_copy_v1.sh
```

Lệnh tự chạy **tuần tự bốn lượt**: no_copy MLP → no_copy GNN → copy MLP → copy GNN. Mỗi lượt seed **42**, batch train **16**, batch validation **32**; bước cache CLIP batch **128**. Tối đa 200 epoch, tối thiểu 30, early stopping patience 30 epoch, validation mỗi 2 epoch. Chọn checkpoint bằng F1 thao tác validation, rồi exact graph, rồi loss.

Output: `runs/level2/magicbrush-easy-copy-v1/summary.json`; checkpoint/prediction ở `<variant>/<model>/42/`. Log: `logs/magicbrush-easy-copy-v1/launcher.log`. Lệnh tự chuyển lượt tiếp theo và dừng sau báo cáo, **không mở test hoặc chạy thêm seed**. Gate chất lượng thấp vẫn được ghi nhận và chạy nốt đối chứng; lỗi số học, hash hoặc runtime sẽ dừng hàng đợi. Chạy lại cùng lệnh để resume checkpoint hoặc xác minh/bỏ qua lượt đã hoàn tất.

Cần đọc F1 từng loại thao tác, exact và tính hợp lệ cùng chẩn đoán tên/ID sau khi chạy; không kết luận cải thiện từ kiểm tra CPU. Các yêu cầu chính của Task B nhiệm vụ 2 vẫn giữ: hơn 300 cặp, split theo ảnh, encoder đóng băng, sáu thao tác, MLP/GNN và chỉ số theo loại. Thực nghiệm chính ba seed và đánh giá test là giai đoạn riêng sau pilot.
