# Kiểm chứng giao diện gán tay MagicBrush v1

Đã triển khai giao diện local tại `tools/magicbrush_manual_v1/`.
[Hướng dẫn sử dụng](../../../docs/LEVEL2_MAGICBRUSH_MANUAL_V1.md).
Không gọi model hoặc GPU, không sửa dữ liệu/split/experiment cũ.

- **280 tests editing CPU PASS**, trong đó **21 tests mới** cho manual tool.
- Browser Chromium tắt GPU kiểm luồng nhập đối tượng/thuộc tính/quan hệ,
  nhập tiếp sau autosave, source-first, giữ ID, diff, hoàn thành và export.
- Kiểm undo/redo xóa vật thể kèm cạnh, mobile không tràn ngang, phục hồi local
  draft/JSON, optimistic conflict hai tab, lịch sử không ghi đè, hash ảnh,
  manifest thay đổi, loader và packet review hiện có.
- Smoke trên manifest thật: **512 train/val** khả dụng, hiển thị đúng **20
  ID khởi động** đã khóa. Ảnh nguồn thật tải được; ảnh đích bị ẩn trước xác
  nhận nguồn; không lỗi JavaScript ở desktop 1512×1080 và mobile 390×844.
- **0 nhãn tay thật được tạo, 0 review thật**. Các thao tác lưu/hoàn thành
  trong tests dùng fixture tổng hợp ở thư mục tạm, tên `UNIT_TEST_ONLY`.
- Export không tự duyệt nhãn hoặc mở training; vẫn cần đủ ≥300 cặp sử dụng
  được, kiểm độc lập ≥20 cặp và protocol/launcher cho annotation manual.

Ảnh xem trước tạo tại máy, không đưa ảnh dataset vào commit:
`data/processed/level2/magicbrush-manual-v1/ui-preview.png` và
`ui-preview-mobile.png`. Các ảnh này là trạng thái chưa gán, không phải
minh chứng manual annotation đã hoàn tất.

```bash
myenv/bin/python -m tools.magicbrush_manual_v1 --port 8776
```

Mở `http://127.0.0.1:8776/`. Dữ liệu tự lưu trong
`data/processed/level2/magicbrush-manual-v1/`; Ctrl+C dừng server. Nếu môi
trường ở máy khác, forward cổng 8776 về máy dùng browser.
