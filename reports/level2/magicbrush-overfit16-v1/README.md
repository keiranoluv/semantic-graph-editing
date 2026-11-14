# MagicBrush: phép thử học thuộc 16 mẫu train, v1

Phép thử này kiểm tra pipeline v3 có học lại chính các mẫu được huấn luyện hay không, trước khi quyết định sửa quy ước nhãn, tối ưu hóa hoặc cách mô hình xác định đối tượng. Đây là chẩn đoán cho Task B, nhiệm vụ 2 trong `Project01_Graph_Editing.pdf`; không thay thế thực nghiệm chính trên ≥300 cặp, ba seed và các tập đánh giá tách biệt.

## Dữ liệu và phạm vi

- Chọn cố định 16 cặp thuộc 16 nhóm nguồn trong 301 mẫu train bằng seed chọn mẫu `20261114`. Ưu tiên phủ loại thao tác hiếm rồi điền đủ; không chọn theo kết quả mô hình. Có đủ sáu loại: 16 add_node, 10 replace_node, 3 delete_node, 17 add_edge, 1 replace_edge, 3 delete_edge.
- `cohort.json` ghi ID, instruction và hash đầu vào/nhãn/thao tác. Kiểm tra không có đầu vào trùng nhau cả ở graph/instruction lẫn tensor đầu vào thực tế.
- Học và chấm trên cùng 16 cặp. Không sử dụng validation cho huấn luyện, chọn checkpoint hoặc chấm probe; không giải mã nhãn test. Preflight cha vẫn đọc/kiểm tra các artifact trainval của v3.
- Giữ nguyên nhãn và toàn bộ từ vựng train v3: 682 nhãn vật thể, 1.448 thuộc tính, 81 quan hệ. Không thu hẹp từ vựng để làm bài thử dễ hơn. Cả 16 mẫu đều biểu diễn được trong từ vựng/budget v3.
- Dùng lại feature CLIP đã đóng băng và kiểm hash từ cache v3. Store trong bộ nhớ của probe chỉ chứa đầu vào của 16 mẫu train và prototype từ vựng train. Không encode CLIP thêm, không nạp checkpoint đã huấn luyện.
- Không sửa source, manifest, config hoặc output của thực nghiệm cũ. Config/lock/provenance của probe độc lập.

## Lệnh chạy GPU dành cho người dùng

Từ repository hiện tại:

```bash
bash scripts/run_magicbrush_overfit16_v1.sh
```

Launcher tự chạy **MLP → GNN**, một model mỗi thời điểm, seed **42**, batch train/eval **16**, khởi tạo mới. Mỗi model có tối đa **3.000 bước cập nhật**, kiểm tra sau mỗi **25 bước**. Giữ kiến trúc, learning rate `0.0003`, loss/class weights và gradient clipping của v3; tắt dropout và weight decay cho phép thử học thuộc.

Dừng sớm nếu **cả output thô và output có ràng buộc** đạt **16/16 exact operation set và exact target graph**, trong **hai lần kiểm tra liên tiếp**. Không dùng patience dừng khi loss chững lại. Nếu MLP hết ngân sách mà chưa học thuộc, launcher vẫn chạy GNN để có đủ hai kết quả chẩn đoán. Sau GNN, launcher dừng; không tự chạy toàn bộ dataset hoặc test.

Chạy lại cùng lệnh để tiếp tục checkpoint gần nhất; có thể tính lại tối đa 24 bước sau checkpoint. Model đã hoàn thành được kiểm hash rồi bỏ qua. Một khóa thiết bị chung ngăn chạy đồng thời với các launcher dùng cùng cơ chế khóa.

## Kết quả cần đọc

- `runs/level2/magicbrush-overfit16-v1/summary.json`: kết luận và metric của cả hai model.
- `<output>/<model>/42/result.json`: lịch sử, checkpoint được chọn, raw/constrained per-type precision/recall/F1, exact match và chẩn đoán các head.
- `<output>/<model>/42/predictions-train.json`: output thô/có ràng buộc từng mẫu.
- `<output>/<model>/42/{last,best}.pt`: checkpoint tiếp tục và checkpoint được chọn.
- `logs/magicbrush-overfit16-v1/launcher.log`: log tích lũy.

`MEMORIZED` nghĩa là đã đạt tiêu chí học thuộc. Khi đó có thể tiếp tục phân tích tính nhất quán của nhãn và khả năng tổng quát hóa; kết quả này không chứng minh nhãn đúng. `NOT_MEMORIZED_WITHIN_BUDGET` nghĩa là chưa đạt trong ngân sách đã đặt, không tự động chứng minh có lỗi code. Khi đó đọc chẩn đoán định vị ID, action, label và attributes trước khi chọn thay đổi tiếp theo.

Các chỉ số head sử dụng mask giám sát tham chiếu chỉ để chẩn đoán; quá trình sinh dự đoán không nhận target. COPY và nhãn literal tương đương có thể khác nhau ở accuracy phân lớp. F1 thuộc tính nhị phân được đo trước bước chọn theo count/top-k, nên không phải F1 thao tác cuối cùng.

## Kiểm tra trước khi bàn giao

`preflight.json` ghi kiểm tra CPU trên dữ liệu thật và hash định danh thực nghiệm. `verification.json` ghi phạm vi kiểm thử CPU. Các test học và resume dùng dữ liệu giả lập nhỏ trên CPU; chưa chạy học/inference GPU trên 16 mẫu thật. Thực nghiệm chính vẫn cần báo cáo riêng ba seed và các metric yêu cầu của đề bài.
