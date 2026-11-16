# Xem lại nhãn không đạt của 200 ảnh Qwen

Theo chỉ dẫn của người dùng ngày 2026-11-16, **mỗi trường trống (JSON null) được hiểu là Đạt**, kể cả mẫu chỉ trống một trường. Giá trị true/false đã chọn giữ nguyên. Đây là quy ước của người gán nhãn, không phải phán xét của assistant hoặc mô hình.

Công cụ độc lập ở `tools/level3_label_recheck/`, không đổi code/config đã khóa của image inference, SGG hoặc evaluation. Không chạy model/GPU, không đọc điểm CLIPScore/LPIPS/graph và chưa chạy báo cáo tương quan.

Sau khi commit code, chuẩn bị một lần:

```bash
bash scripts/review_level3_labels.sh prepare \
  --review qwen200-reviewed.json --blank-is-pass
```

Đầu vào phải chứa đầy đủ 200 task ID, đúng cohort và image binding. Chỉ null hiện diện rõ ràng mới được mặc định; file thiếu dòng/trường, ID trùng, nhãn kiểu số hoặc binding sai đều bị từ chối. Thiếu cờ `--blank-is-pass` thì không tự áp dụng quy ước cho file có null.

Output: `runs/level3/evaluation-v1/seed42/label-recheck-v1/`:

- `original-reviewed.json`: bản sao nguyên byte của file người dùng; file gốc không sửa.
- `normalized-reviewed.json`: đủ 200 nhãn, kèm hash file gốc và danh sách từng trường được mặc định theo quy ước.
- `comparison.html`: chỉ các mẫu có ít nhất một tiêu chí false, ảnh nguồn/đích, instruction và hai trường nhãn.
- `summary.json`, `receipt.json`: thống kê, hash và code commit. Không ghi đè output đã tồn tại; lượt chuẩn bị mới dùng `OUTPUT=runs/level3/evaluation-v1/seed42/<ten-moi>`.

Mở trang:

```bash
bash scripts/review_level3_labels.sh serve
# http://localhost:8010/comparison.html
```

Trang giữ nguyên danh sách mẫu cần kiểm tra kể cả sau khi sửa thành Đạt/Đạt, để mẫu không biến mất khi đang duyệt. Không hiển thị điểm tự động hoặc trường ghi chú. Nhãn sửa lưu ở localStorage riêng theo hash file đầu vào, không đụng nhãn của giao diện 200 ảnh ban đầu.

Sau khi xem lại, bấm **Tải nhãn 200 mẫu**, nhận `qwen200-rechecked.json`. File này chứa toàn bộ 200 nhãn, giữ những mẫu không thuộc danh sách lỗi, ghi lại thay đổi và quy ước ban đầu. Có thể nhập lại vào chính trang xem lại để tiếp tục; không nhận file khác bộ ảnh hoặc file sửa các mẫu ngoài danh sách của lượt này. Không gọi những mẫu không sửa là đã được duyệt lại.

Gửi lại file đã chốt trước khi chạy báo cáo tương quan cuối cùng. File xuất tương thích với `bash scripts/run_level3_eval.sh report --review ...`; không cần chạy lại ảnh, SGG hoặc metric.

Kiểm tra CPU:

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-evaluation-v1/venv/bin/python -m pytest -q \
  tests/level3/test_label_recheck.py tests/level3/test_evaluation200.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/level3/test_label_recheck_browser.py
```

Browser test dùng Playwright/Chromium trong context độc lập, kiểm tra sửa nhãn, reload, export đủ 200 dòng, giữ những mẫu khác, round-trip và từ chối binding sai.
