# Duyệt Qwen với hai trường

> Tài liệu thực nghiệm lịch sử. Launcher/giao diện pilot tương ứng đã nghỉ dùng; các kết quả ảnh/graph được giữ. Dùng [luồng hiện tại](LEVEL3_CLEANUP.md); code cũ có thể đọc lại tại commit `1d2d876`.

Giao diện chỉ hiển thị ảnh nguồn, ảnh Qwen, câu lệnh và hai lựa chọn **Đúng yêu cầu** / **Bảo toàn**. Không có cột Pix2Pix, ô ghi chú hoặc tên người duyệt. Chọn Đạt, Không đạt hoặc để Chưa chấm; xuất JSON không yêu cầu ghi chú. Hai trường đều có giá trị boolean thì trạng thái là `reviewed`, kể cả khi chọn Không đạt; còn thiếu trường thì `pending`.

Trang mới đọc outputs cũ trên CPU, không inference lại, không sửa/xóa ảnh hoặc receipt đã khóa. Mỗi bản giao diện được ghi vào thư mục phiên bản mới. Cả 100 cặp và pilot 6 cặp dùng chung renderer. Preview RGBA ghép nền trắng như trước.

## Tạo trang

Chạy từ root dự án với env có Pillow:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_qwen_review_v2 \
  --run runs/level3/qwen100-v1/seed42-v1 \
  --output runs/level3/qwen-review-v2/qwen100-v1

CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level3_qwen_review_v2 \
  --run runs/level3/qwen21-v1/comparison-v1 \
  --review qwen21-reviewed.json \
  --output runs/level3/qwen-review-v2/pilot-v1
```

Đã tạo hai output trên; lệnh từ chối ghi đè nếu thư mục tồn tại. `--review` không bắt buộc; khi dùng, kiểm tra `results_sha256`, chỉ lấy hai nhãn của Qwen. File đánh giá cũ giữ nguyên. Pilot hiện kế thừa đủ 6 đánh giá Qwen đã lưu, gồm một giá trị Bảo toàn = false; bộ 100 chưa có nhãn ban đầu.

## Mở giao diện

```bash
python3 -m http.server 8004 --bind 127.0.0.1 \
  --directory runs/level3/qwen-review-v2/qwen100-v1
# http://localhost:8004/comparison.html
```

Pilot dùng cổng 8003 và thư mục `runs/level3/qwen-review-v2/pilot-v1`. Hai địa chỉ cũ được giữ khi chuyển server sang thư mục mới; tải lại trang để dùng giao diện mới. Nếu server đang chạy trên cổng đó, không khởi động thêm bản thứ hai.

Lựa chọn tự lưu trong localStorage của trình duyệt, gắn với hash kết quả. Đổi trang hoặc tải lại giữ các lựa chọn; trình duyệt/địa chỉ khác không tự chia sẻ dữ liệu. Nút **Tải kết quả JSON** xuất `qwen100-reviewed-v2.json` hoặc `qwen21-reviewed-v2.json`; nên tải để lưu lâu dài. Nếu trình duyệt chặn lưu localStorage, trang hiển thị thông báo. Không tự gán tên reviewer hoặc chấm điểm chất lượng.

JSON dùng schema `qwen-only-review-v2`, giữ `task_id`, `model=qwen21`, `status`, hai trường boolean/null, `results_sha256` và hash receipt đầu vào. Không có trường notes/reviewer; bên đọc nhãn sau này cần hỗ trợ schema này, không áp điều kiện ghi chú/tên của UI cũ.

## Kiểm tra CPU thực tế

Đã kiểm tra Chromium headless với `--disable-gpu`: 106 mẫu, 212 preview giải mã được, đúng 2 ô chấm mỗi mẫu, không Pix2Pix/ghi chú/tên người duyệt; xuất JSON không cần nhập thêm. Lựa chọn false được giữ nguyên; chấm một phần hoặc xóa một lựa chọn trả về pending; chuyển trang và reload giữ lựa chọn. Không có lỗi JavaScript. Kiểm tra toàn bộ receipt của hai run cũ và hai trang mới thành công.

Bằng chứng: `runs/level3/qwen-review-v2/validation-v2/`, gồm `validation.json` và screenshot hai trang. Các thao tác chọn trong browser test là dữ liệu thử trong browser context riêng, không phải nhãn người dùng. Không chạy GPU.
