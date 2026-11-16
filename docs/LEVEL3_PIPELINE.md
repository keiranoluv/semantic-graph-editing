# Level 3: hai bước độc lập

Một bộ dữ liệu: **100 ảnh gốc, 200 câu edit, 200 ảnh edited và 300 scene graph**. Mỗi ảnh gốc có hai câu edit độc lập; cả hai đều chỉnh từ ảnh gốc.

Chạy từng bước khi cần:

```bash
# Bước 1: Qwen tạo 200 ảnh edited
bash scripts/run_level3_qwen.sh

# Bước 2: REACT++ tạo graph cho 100 ảnh gốc + 200 ảnh của bước 1
bash scripts/run_level3_sgg.sh
```

Mỗi lệnh chạy phần còn thiếu của lượt hiện tại rồi dừng. Không có bước tự gọi bước khác hoặc hàng đợi model. Bước 2 yêu cầu bước 1 có đủ 200 ảnh, nhưng có thể chạy ở thời điểm khác. Hiện lượt `seed42` đã đủ kết quả nên hai lệnh chỉ xác minh và resume, không infer lại.

Muốn tái lập từ đầu, chọn cùng một tên lượt mới cho hai bước:

```bash
RUN_ID=reproduce01 bash scripts/run_level3_qwen.sh
RUN_ID=reproduce01 bash scripts/run_level3_sgg.sh
```

- Ảnh của lượt này: `runs/level3/qwen200/reproduce01/`.
- Graph của lượt này: `runs/level3/qwen-sgg/reproduce01/`.
- SGG tự đọc thư mục Qwen cùng tên. Không cần truyền đường dẫn input, tạo manifest/config mới hoặc chạy lệnh nối hai bước.
- Lặp lại cùng lệnh/cùng `RUN_ID` để resume; chỉ đổi tên khi muốn một lượt mới. Kết quả có receipt trong lượt đó được giữ. Không nhập kết quả của lượt cũ vào một lượt mới.

Qwen dùng checkpoint chính thức Qwen-Image-2.1, BF16, batch 1, seed 42, 40 steps, CPU offload. REACT++ dùng checkpoint/decoder ONNX đã khóa, batch 1, seed 42. Hai môi trường CUDA 12.8/RTX 5090 được giữ; inference có W&B theo dõi. Không huấn luyện. Lượt mới có provenance và hash ảnh riêng, không buộc ảnh mới phải có cùng byte với lượt cũ.

Xem ảnh hoặc graph bằng cách thay lệnh mặc định thành `serve`:

```bash
bash scripts/run_level3_qwen.sh serve  # http://localhost:8008/comparison.html
bash scripts/run_level3_sgg.sh serve   # http://localhost:8009/comparison.html
```

Có thể thêm cùng `RUN_ID=...` để xem lượt khác. Server chỉ dựng trang trên CPU. UI/ảnh/graph hiện tại và các thay đổi riêng của người dùng được giữ nguyên.

Chi tiết thực nghiệm: [Qwen](LEVEL3_QWEN.md), [SGG](LEVEL3_SGG.md), [dọn dẹp và artifacts được giữ](LEVEL3_CLEANUP.md).

Sau khi có ảnh/SG, dùng [bộ đánh giá 200 cặp](LEVEL3_EVALUATION.md) để chấm graph trên CPU, tính CLIPScore/LPIPS và đối chiếu nhãn tay. Đây là bước độc lập, không được hai launcher trên tự gọi.

Bản bàn giao gồm [gói sao lưu và notebook demo](LEVEL3_SUBMISSION.md), [data card](LEVEL3_DATA_CARD.md) và [báo cáo Task 3](LEVEL3_TASK3_REPORT.md). Notebook đọc kết quả đã lưu, tính lại điểm graph trên CPU và không khởi chạy inference.

Kiểm tra CPU: 48 tests của Qwen/SGG và chuỗi hai bước đều qua; bao gồm 200 case tổng hợp, 300 prediction tổng hợp, ngắt/resume và từ chối input sai. Đã đọc kiểm tra 200 cặp/300 ảnh thật bằng tên lượt khác qua bản sao metadata tạm, xác nhận cùng model protocol và không đổi artifact cũ. Không chạy thêm inference ảnh hoặc SGG thật trong lần sửa này.
