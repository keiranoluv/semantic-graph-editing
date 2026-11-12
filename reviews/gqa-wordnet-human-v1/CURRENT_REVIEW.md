# Bản review đang hiển thị

Viewer hiện tự nạp bản
[`review-ce1dcb634ff05050.csv`](submissions/review-ce1dcb634ff05050.csv):
**30/30 nhãn đã có quyết định, 23 accept và 7 defer**. Bảy defer được ghi theo
xác nhận rõ ràng của người dùng trong cuộc trò chuyện; bản CSV gốc giữ nguyên.
Xem [biên bản quyết định](submissions/review-ce1dcb634ff05050.validation.json).

Chỉ cần tải lại trang đang mở tại `http://127.0.0.1:8765/`, hoặc mở lại
`index.html`. Không cần chọn file để nhập thủ công. Server đang chạy đọc lại
HTML ở mỗi request nên không cần khởi động lại.

Lần đầu nạp snapshot này, viewer sao lưu bản nháp localStorage hiện có rồi
nhập CSV bằng chính bộ kiểm tra import của giao diện. Snapshot chỉ được nhập
một lần trên mỗi trình duyệt; các chỉnh sửa mới sau đó được giữ khi reload.
Trình duyệt mới hoặc đã xóa localStorage sẽ nạp lại bản review đã chốt.

Bản nháp cũ được lưu dưới key
`gqa-wordnet-review:<packet_sha256>:before-snapshot:<csv_sha256>` trong
localStorage, nếu trình duyệt cho phép lưu. Khi localStorage bị chặn, CSV vẫn
được nạp vào phiên hiện tại; nên xuất CSV để giữ chỉnh sửa mới.

Để tái áp dụng bản này vào viewer:

```bash
myenv/bin/python -m tools.apply_gqa_review_snapshot \
  --csv reviews/gqa-wordnet-human-v1/submissions/review-ce1dcb634ff05050.csv
```

Lệnh xác minh toàn bộ 30 ID/nhãn, metadata gốc, ứng viên, người duyệt, quyết
định và fingerprint bằng chứng trước khi thay HTML. Chạy lại cùng lệnh là
idempotent. Dữ liệu train, mapping, KG, embedding, checkpoint và bộ nguồn
viewer ban đầu giữ nguyên; không chạy training.

[`index-before-confirmed-review.html`](index-before-confirmed-review.html)
giữ nguyên HTML trước cập nhật, khớp hash trong `verification.json` ban đầu.
File `index.html` hiện tại chỉ được bổ sung khối nhập snapshot; packet
`review-data.json` và source CSV không đổi. Không chạy lại lệnh **build** vào
thư mục này để thay HTML đã có review; workflow build giữ nguyên cơ chế từ
chối ghi đè khác nội dung. Dùng thư mục mới nếu cần tạo lại bộ bằng chứng.

[Snapshot có provenance](review-snapshot-ce1dcb634ff05050.json) và
[biên bản cập nhật viewer](viewer-update-ce1dcb634ff05050.json) lưu checksum
nguồn và HTML mới. Việc nạp snapshot chỉ cập nhật quyết định trong giao diện;
chưa áp dụng các lựa chọn đó vào mapping dùng cho mô hình.
