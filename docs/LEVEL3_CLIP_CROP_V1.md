# NV3 · Ảnh hưởng của center crop đến CLIPScore

Cùng 200 ảnh đích VN100, caption tiếng Anh, tiền tố và CLIP ViT-B/32 đã khóa trong [protocol đánh giá](../configs/level3/evaluation200-v1.json). Mỗi ảnh được chấm hai lần liên tiếp. Control dùng CLIPProcessor gốc. Letterbox resize bicubic cạnh dài về 224 px, giữ tỉ lệ rồi thêm viền RGB (123, 117, 104) đến 224 × 224. Mỗi điểm control được đối chiếu với điểm cũ, dung sai tuyệt đối 0,0001. Hai cách tiền xử lý cũng khác về kích thước hiển thị của vật thể.

Từ gốc repo, trên môi trường đánh giá NV3 đã có checkpoint và đã đăng nhập W&B:

```bash
bash scripts/run_level3_clip_crop_v1.sh
```

Inference trên một GPU, FP32, batch 1, seed 42. Launcher chấm đủ hai nhánh trên 200 cặp rồi tổng hợp trên CPU và dừng. Terminal và W&B ghi tiến độ sau cặp đầu tiên và mỗi 10 cặp. ETA chỉ tính phần inference. Log mỗi lần chạy nằm trong `runs/level3/clip-crop-v1/seed42/logs/`. Bảng số liệu theo cặp và tổng hợp lần lượt là `scores.json` và `summary.json` trong thư mục `cuda/` bên cạnh. Khi chạy lại, launcher tính lại từ đầu.

Chỉ số chính là Spearman với nhãn tay đạt cả yêu cầu chỉnh sửa và bảo toàn trên 200 cặp. Phân tích phụ dùng nhãn thực hiện yêu cầu, nhóm có điểm đồ thị, và hai nhóm theo vị trí vùng chỉnh sửa. Nhóm sát mép có ít nhất 25% diện tích ROI yêu cầu nằm ngoài center crop, tính trước từ ROI nguồn chuẩn hóa chiếu lên kích thước ảnh đích. Đây là xấp xỉ vị trí chỉnh sửa. Các chênh lệch tương quan dùng 2.000 bootstrap ghép cặp theo ảnh nguồn. Mẫu bootstrap có vector hằng được bỏ riêng và báo số lượt hợp lệ. Ca 58 đã được xem khi hình thành giả thuyết. Kết quả mô tả tập phát triển VN100 này.

Kiểm tra CPU trên ba ảnh thật (48, 58, 168) và các fixture hình học/thống kê:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_clip_crop_v1.sh --device cpu --smoke
CUDA_VISIBLE_DEVICES='' data/cache/level3-evaluation-v1/venv/bin/python -m pytest -q tests/level3/test_clip_crop_v1.py
```
