# Pilot liên kết câu yêu cầu với nhãn và ID nguồn

Thử một thay đổi hẹp trên baseline compact: bổ sung tín hiệu khớp từ trong câu yêu cầu để hỗ trợ chọn nhãn đối tượng và action trên ID nguồn. Giữ nguyên 301 train, 67 validation (64 có thao tác), nhãn, từ vựng, cache CLIP, seed, kiến trúc nền, optimizer, cách chọn checkpoint và chấm điểm. Không đọc nhãn test.

## Cơ chế

Mô hình thêm **72 tham số học** dưới dạng các lượng cộng vào logits. Đặc trưng chỉ nhận graph nguồn và instruction:

- Nhãn: khớp cả cụm từ, khớp từ cuối, tên chưa có trong nguồn, tên mới xuất hiện đầu tiên và vị trí sau từ nối thay thế. Chuẩn hóa một số dạng số nhiều; không dùng từ điển đồng nghĩa lấy từ validation/test.
- ID nguồn: tên node được nhắc, thuộc tính được nhắc, số node cùng khớp tên, cosine CLIP và kết hợp với từ gợi ý thêm/xóa/thay đổi. Cạnh nhận tín hiệu từ hai đầu mút và predicate.
- Đây là ưu tiên mềm, không ép chọn tên được nhắc, không ép action theo từ khóa và không lọc bỏ các nhãn khác. Các trọng số được học cùng model.

Khởi tạo trọng số label mới theo thứ tự trong preflight: `[4, 0.5, 1, 1, 1, -1]`; label thay thế `[0, 0, 2, 1, 2, 0]` chỉ kích hoạt với từ gợi ý thay thế. Action node ban đầu cộng 2 cho xóa khi khớp tên + từ gợi ý xóa, cộng 2 cho thay khi khớp tên + từ gợi ý thay. Các trọng số action còn lại khởi tạo 0. Đây là prior từ vựng được khai báo, không phải kết quả học tự phát. Mô hình vẫn có thể học điều chỉnh chúng.

**Giới hạn:** vẫn có 682 nhãn đối tượng train; đây là cơ chế chọn tên được nhắc, chưa phải bộ sao chép tên mới ngoài từ vựng. Vẫn còn bài toán đại từ, đồng nghĩa, nhiều instance cùng tên và thay đổi phụ không nói trong yêu cầu. Không thay đổi annotation để khớp dự đoán.

## Chạy và ngân sách

```bash
bash scripts/run_magicbrush_lexical_pilot_v1.sh
```

Dùng lại cache CLIP đóng băng của baseline, không encode CLIP lại. **MLP seed 42 → GNN seed 42** tuần tự; batch train **16**, validation **32**, learning rate `0.0003`, dropout `0.1`, weight decay `0.01`. Tối đa **200 epoch/model**, minimum 30, early stopping patience 30 epoch, đánh giá mỗi hai epoch. Khởi tạo mới, không nạp trọng số đã huấn luyện của baseline.

Sau mỗi model, lưu dự đoán 67 validation và chạy ba phép kiểm tra tại checkpoint đã chọn: tắt phần nhãn, tắt phần ID, tắt cả hai. Đây là **ablation chỉ lúc suy luận**, không phải ba model được huấn luyện lại. Xong hai model và các phép kiểm tra thì dừng; không tự thêm seed hoặc test. Chạy lại cùng lệnh để tiếp tục checkpoint và kiểm hash các model đã hoàn thành.

Output: `runs/level2/magicbrush-lexical-pilot-v1/summary.json`; từng model có `validation-report.json`, `ablation.json`, dự đoán và checkpoint. Log: `logs/magicbrush-lexical-pilot-v1/launcher.log`.

## Phạm vi đánh giá và quyền chạy

Chấm chính trên cùng 64 validation có thao tác như baseline; giữ trường hợp ngoài từ vựng trong mẫu số. Ba mẫu ngoài schema vẫn được lưu dự đoán, không tính dự đoán rỗng là đáp ứng yêu cầu. Báo cáo thêm exact bảo thủ chia 67. Chỉ một seed nên chưa kết luận ưu thế thống kê hoặc thay thế thực nghiệm chính ba seed của nhiệm vụ 2.

Người dùng đã cho phép riêng lần này: “Oke cài đặt, lần này bạn có thể tự chạy và phân tích kết quả.” Vì vậy assistant được tự chạy pilot sau khi kiểm tra CPU và commit, thay cho quy ước bàn giao GPU thông thường. Các phiên bản cũ và notebook cá nhân không đổi.

`preflight.json`, `cpu-check.json`, `verification.json` là chứng cứ trước chạy GPU. Test kiểm tra khớp từ theo ranh giới, ID/thuộc tính nguồn, không nhận target, cùng khởi tạo và logits với baseline khi tắt residual, học trên dữ liệu giả lập, queue/resume và hash. Kết quả GPU sau chạy sẽ được phân tích riêng; không dùng kết quả test để điều chỉnh phiên bản này.
